from argparse import ArgumentParser

import numpy as np
import wandb


def main(args):

    args.dst_entity = args.dst_entity if args.dst_entity is not None else args.src_entity
    args.dst_project = args.dst_project if args.dst_project is not None else args.src_project
    same_project = args.src_entity == args.dst_entity and args.src_project == args.dst_project
    name_append = "-extended" if same_project and args.name is None else ""

    # Set your API key
    wandb.login()
    # Initialize the wandb API
    api = wandb.Api()
    
    # Get the runs from the source project
    runs = api.runs(f"{args.src_entity}/{args.src_project}")
    
    # Iterate through the runs and copy them to the destination project
    selected_runs = {}
    for run in runs:
        if run.id in {args.run1, args.run2}:
            selected_runs[run.id] = run

    name = selected_runs[args.run1].name + name_append if args.name is None else args.name
    # Create a new run in the destination project
    new_run = wandb.init(
        project=args.dst_project, entity=args.dst_entity, config=selected_runs[args.run1].config, name=name, resume="allow"
    )

    for run_id in [args.run1, args.run2]:

        run = selected_runs[run_id]
        # Log the history to the new run
        history = run.history(samples=run.lastHistoryStep + 1)
        for index, row in history.iterrows():
            new_run.log({k: v for k, v in row.to_dict().items() if not np.isnan(v)})

        # Upload the files to the new run
        files = selected_runs[run_id].files()
        for file in files:
            file.download(replace=True)
            new_run.save(file.name, policy="now")

    # Finish the new run
    new_run.finish()


if __name__ == "__main__":
    parser = ArgumentParser(description="Extend wandb run1 with run2")
    parser.add_argument("-se", "--src-entity", type=str, default="indezera", help="Source wandb entity name.")
    parser.add_argument("-sp", "--src-project", type=str, help="Name of the wandb project.")
    parser.add_argument("-de", "--dst-entity", type=str, default=None, help="Destination wandb entity name.")
    parser.add_argument("-dp", "--dst-project", type=str, default=None, help="Name of destination wandb project.")
    parser.add_argument("-r1", "--run1", type=str, help="ID of the base run.", required=True)
    parser.add_argument("-r2", "--run2", type=str, help="ID of the run to append.", required=True)
    parser.add_argument("-n", "--name", type=str, help="Name for the new combined run.", required=True)

    main(parser.parse_args())
