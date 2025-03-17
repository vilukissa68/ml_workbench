#!/usr/bin/env python3
import torch.optim as optim


def get_optimizer(model, args):
    # Ensure both betas are set
    if args.optimizer_beta1 and args.optimizer_bet2:
        betas = (args.optimizer_beta1, args.optimizer_beta2)
    else:
        betas = (0.9, 0.999)

    if args.optimizer == "SGD":
        optimizer = optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=0.9,
            weight_decay=args.weight_decay,
            betas=betas,
        )
    elif args.optimizer == "Adam":
        # optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=betas)
        optimizer = optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
    elif args.optimizer == "AdamW":
        optimizer = optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=betas
        )
    else:
        raise ValueError(f"Optimizer {args.optimizer} is not supported!")

    if args.scheduler == "StepLR":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=args.scheduler_step_size, gamma=args.scheduler_gamma
        )
    elif args.scheduler == "MultiStepLR":
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[5, 10, 15], gamma=args.scheduler_gamma
        )
    elif args.scheduler == "ReduceLROnPlateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.1, patience=10, verbose=True
        )
    # elif args.scheduler == "LambdaLR":
    #     scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=schedule_func)
    else:
        print("No scheduler")
        scheduler = None

    return optimizer, scheduler
