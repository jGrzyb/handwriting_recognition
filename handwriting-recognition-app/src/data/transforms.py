from torchvision import transforms

def get_transforms(augment=False):
    if augment:
        return transforms.Compose([
            transforms.RandomRotation(15, expand=True, fill=(255,)),
            transforms.RandomAffine(0, translate=(0.05, 0.05), fill=(255,)),
            transforms.RandomPerspective(distortion_scale=0.5, p=0.5, fill=(255,)),
            transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        ])
    
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((64, 128)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: 1.0 - x),
        transforms.Normalize((0.5,), (0.5,)),
    ])