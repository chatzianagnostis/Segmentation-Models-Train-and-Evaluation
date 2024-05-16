import segmentation_models_pytorch as smp


class CustomSegmentationModel:
    def __init__(self, model_type, **kwargs):
        self.model_type = model_type
        self.kwargs = kwargs

    def build_model(self):
        if self.model_type == 'Unet':
            return smp.Unet(
                **self.kwargs
            )
        elif self.model_type == 'UnetPlusPlus':
            return smp.UnetPlusPlus(
                **self.kwargs
            )
        elif self.model_type == 'MAnet':
            return smp.MAnet(
                **self.kwargs
            )
        elif self.model_type == 'Linknet':
            return smp.Linknet(
                **self.kwargs
            )
        elif self.model_type == 'FPN':
            return smp.FPN(
                **self.kwargs
                    )
        elif self.model_type == 'PSPNet':
            return smp.PSPNet(
                **self.kwargs
                    )
        elif self.model_type == 'PAN':
            return smp.PAN(
                **self.kwargs
                    )
        elif self.model_type == 'DeepLabV3':
            return smp.DeepLabV3(
                **self.kwargs
                    )
        elif self.model_type == 'DeepLabV3Plus':
            return smp.DeepLabV3Plus(
                **self.kwargs
                    )
                    