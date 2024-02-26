import torchvision.transforms as T


def get_transforms(args):
    if args.dataset == "splitcifar100":
        train_transforms = T.Compose(
            [
                T.ToTensor(),
                T.RandomResizedCrop(224, interpolation=T.InterpolationMode.BILINEAR),
                T.RandomHorizontalFlip(),
            ]
        )

        test_transforms = T.Compose(
            [
                T.ToTensor(),
                T.Resize(256, interpolation=T.InterpolationMode.BILINEAR),
                T.CenterCrop(224),
            ]
        )

    elif args.dataset == "splittinyimagenet":
        train_transforms = T.Compose(
            [
                T.ToTensor(),
                T.RandomResizedCrop(224, interpolation=T.InterpolationMode.BILINEAR),
                T.RandomHorizontalFlip(),
            ]
        )

        test_transforms = T.Compose(
            [
                T.ToTensor(),
                T.Resize(256, interpolation=T.InterpolationMode.BILINEAR),
                T.CenterCrop(224),
            ]
        )

    elif args.dataset == "splitcifar10":
        train_transforms = T.Compose(
            [
                T.ToTensor(),
                T.RandomResizedCrop(224, interpolation=T.InterpolationMode.BILINEAR),
                T.RandomHorizontalFlip(),
            ]
        )

        test_transforms = T.Compose(
            [
                T.ToTensor(),
                T.Resize(256, interpolation=T.InterpolationMode.BILINEAR),
                T.CenterCrop(224),
            ]
        )


    return train_transforms, test_transforms
