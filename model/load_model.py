import os
import model.model_pt_architecture as tm

service_path = "model"

def model_pt_res18():
    """
    Load Resnet18 model from path
    """
    model = tm.resnet18(modelPath=os.path.join(service_path, 'model_pt_res18.pth'))
    return model


if __name__ == '__main__':
    main()