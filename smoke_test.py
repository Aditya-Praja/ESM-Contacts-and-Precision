import esm, torch

model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
model.eval().to("mps")
batch_converter = alphabet.get_batch_converter()

ubq = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
_, _, tokens = batch_converter([("ubq", ubq)])

with torch.no_grad():
    output = model(tokens.to("mps"), repr_layers=[12], return_contacts=True)
    
print(output["contacts"].shape)
print(output["representations"][12].shape)