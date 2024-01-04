import torch
from torchvision.transforms import InterpolationMode, functional as TF
from torchvision.transforms.functional import to_tensor
from PIL import Image
from components.physics import get_physics
from components.SwinIR import get_model




device = 'cuda' if torch.cuda.is_avaliable() else 'cpu'


class SwinPipeline:

    def __init__(self, image, device, resize) -> None:
        self.device = device
        self.image = image
        self.resize = resize

    def main(self):

        y = self._read_image()
        model = self._load_model()
        
        model.eval()
        with torch.inference_mode():
            x = model(y)

        x = x.squeeze().permute(1, 2, 0).cpu()

        return x


    def _read_image(self):

        y = Image.open(self.image)
        y = to_tensor(y)
        y = y.to(device)

        if self.resize is not None:
            y = TF.resize(
                y,
                size=self.resize,
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            )

        return y
    

    def _load_model(self):

        model = get_model('deblurring')
        model.to(device)

        weights_url = f"https://huggingface.co/jscanvic/scale-equivariant-imaging/resolve/main/Deblurring_Gaussian_R2_Noise5_Proposed.pt?download=true"
        weights = torch.hub.load_state_dict_from_url(
            weights_url, map_location='cpu'
        )

        model.load_state_dict(weights)

        return model



def swin_api(image, device= device):

    pipeline = SwinPipeline(
        image= image,
        device= device,
        resize= 256
    )

    y = pipeline.main()

    return y

