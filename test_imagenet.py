import torch, cv2, os
from resnet_imagenet import resnet50, GroupableConv2d
from utils import update_permutation_matrix, impose_group_lasso

if not os.path.exists("./test/before"):
    os.makedirs("./test/before")
if not os.path.exists("./test/after"):
    os.makedirs("./test/after")

model = resnet50(group1x1=True).cuda()
update_permutation_matrix(model, 1)

optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)

for name, m in model.named_modules():
    if isinstance(m, GroupableConv2d):
        weight_norm = torch.norm(m.weight.data.view(m.out_channels, m.in_channels, -1), p=1, dim=-1).detach().cpu().numpy()
        shuffled_weight_norm = weight_norm[m.P, :][:, m.Q]
        shuffled_weight_norm /= np.max(shuffled_weight_norm) + 1e-8
        cv2.imwrite("./test/before/%s.png" % name, shuffled_weight_norm)

for i in range(100):
    x = torch.randn(2, 3, 64, 64).cuda()
    y = torch.mean(model(x))
    y.backward(torch.tensor([0.]).cuda())
    impose_group_lasso(model, 5e-5)
    optimizer.step()

for name, m in model.named_modules():
    if isinstance(m, GroupableConv2d):
        weight_norm = torch.norm(m.weight.data.view(m.out_channels, m.in_chennels, -1), p=1, dim=-1).detach().cpu().numpy()
        shuffled_weight_norm = weight_norm[m.P, :][:, m.Q]
        shuffled_weight_norm /= np.max(shuffled_weight_norm) + 1e-8
        cv2.imwrite("./test/after/%s.png" % name, shuffled_weight_norm)
