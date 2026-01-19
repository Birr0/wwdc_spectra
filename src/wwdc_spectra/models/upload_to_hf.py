import os

import torch
from dotenv import load_dotenv
from flow import VelocityField

load_dotenv()

data_name = "sdss_II"
experiment_name = "spender_I_flow"
model_names = [
    f"6997867_{i}.ckpt" for i in range(3)
]
data_root = os.getenv("DATA_ROOT")

def ckpt_path(
    experiment_name,
    model_name
):
    return f"{data_root}/{data_name}/{experiment_name}/ckpts/{model_name}"

code_dim=10
hidden_dim=256 
cond_dim = 1
model = VelocityField(
    code_dim, 
    hidden_dim, 
    cond_dim
)

for i in range(3):
    ckpt = torch.load(
        ckpt_path(
            experiment_name,
            model_names[0]
        )
    )

    state_dict = {}
    for k in list(ckpt["state_dict"].keys()):
        prefix, key = k.split(".", 1)
        if prefix == "vf":
            state_dict[key] = ckpt["state_dict"][f"{prefix}.{key}"]

    model.load_state_dict(
        state_dict
    )
    model.push_to_hub(f"Birr001/spender-I-vf-{i}")

# could possibly clean this up using config loaders.


'''class MyModel(
    nn.Module,
    PyTorchModelHubMixin, 
    # optionally, you can add metadata which gets pushed to the model card
    repo_url="your-repo-url",
    pipeline_tag="text-to-image",
    license="mit",
):
    def __init__(self, num_channels: int, hidden_size: int, num_classes: int):
        super().__init__()
        self.param = nn.Parameter(torch.rand(num_channels, hidden_size))
        self.linear = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        return self.linear(x + self.param)

# create model
config = {"num_channels": 3, "hidden_size": 32, "num_classes": 10}
model = MyModel(**config)

# save locally
#model.save_pretrained("my-awesome-model")

login()
# push to the hub
model.push_to_hub("Birr001/test")

# reload
model = MyModel.from_pretrained("Birr001/test")
'''

