import torch
import torch.nn.functional as F
import torch.optim as optim

from data_loader import load_data
from plots import plot_MNIST_data, plot_loss, plot_split
from create_data import create_labeled_data, create_unlabeled_data, create_testing_data
from VAT import VAT
from net import Net
from metric import ContrastiveLoss
from classifier import test_classification

n_epochs = 30
batch_size = 32
alpha = 0.2
beta = 0.0001

random_seed = 0
torch.manual_seed(random_seed)
torch.backends.cudnn.enabled = True

# Load the data
train_loader, test_loader = load_data()

# Training data
train_set = enumerate(train_loader)
train_batch_idx, (train_data, train_targets) = next(train_set)

# Testing data
test_set = enumerate(test_loader)
test_batch_idx, (test_data, test_targets) = next(test_set)

# Vizualize MNIST dataset
plot_MNIST_data(train_data, train_targets)

# Create labeled, unlabeled and test data
n = 100
labeled_pos, labeled_neg = create_labeled_data(n, train_targets)
unlabeled = create_unlabeled_data(6000)
test = create_testing_data(test_targets)


def get_accuracy(test, threshold):
    correct = 0
    for i in range(len(test)):
        img0 = test_data[test[i][0]].unsqueeze(0)
        img1 = test_data[test[i][1]].unsqueeze(0)
        true_label = test[i][2]
        img0, img1 = img0.cuda(), img1.cuda()

        output_f1 = net(img0)
        output_f2 = net(img1)

        euclidean_distance = F.pairwise_distance(output_f1, output_f2)

        if euclidean_distance > threshold and true_label == -1:
            correct += 1
        if euclidean_distance <= threshold and true_label == 1:
            correct += 1
    return correct / len(test)


net = Net().cuda()
criterion = ContrastiveLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)
vat = VAT(net)

iteration = 0
counter = []
loss_history = []
loss_history_d = []
loss_history_g = []
loss_history_step = []

train_acc = []
for epoch in range(0, n_epochs):
    net.train()
    train_loss = 0.0
    train_d_loss = 0.0
    train_g_loss = 0.0
    for i in range(int(len(unlabeled) / 16)):
        f1 = [train_data[labeled_pos[(i * 8 + j) % len(labeled_pos)][0]] for j in range(8)]
        f1 += [train_data[labeled_neg[(i * 8 + j) % len(labeled_neg)][0]] for j in range(8)]
        f1 += [train_data[unlabeled[i * 16 + j][0]] for j in range(16)]
        img0 = torch.stack(f1, 0)
        f2 = [train_data[labeled_pos[(i * 8 + j) % len(labeled_pos)][1]] for j in range(8)]
        f2 += [train_data[labeled_neg[(i * 8 + j) % len(labeled_neg)][1]] for j in range(8)]
        f2 += [train_data[unlabeled[i * 16 + j][1]] for j in range(16)]
        img1 = torch.stack(f2, 0)
        f3 = [labeled_pos[(i * 8 + j) % len(labeled_pos)][2] for j in range(8)]
        f3 += [labeled_neg[(i * 8 + j) % len(labeled_neg)][2] for j in range(8)]
        f3 += [unlabeled[i * 16 + j][2] for j in range(16)]
        label = torch.FloatTensor(f3)

        img0, img1, label = img0.cuda(), img1.cuda(), label.cuda()

        optimizer.zero_grad()
        output1 = net(img0)
        output2 = net(img1)
        loss_d = criterion(output1, output2, label)

        loss_vat = vat(img0, output1) + vat(img1, output2)

        reg = 0
        for p in net.parameters():
            reg = reg + torch.norm(p)

        loss = loss_d + alpha * loss_vat + beta * reg
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_d_loss += loss_d
        train_g_loss += loss_vat

        if iteration % 100 == 0:
            counter.append(iteration)
            loss_history_step.append(loss.item())
        iteration += 1
    net.eval()
    train_acc.append(get_accuracy(test, 0.8))

    loss_history.append(train_loss)
    loss_history_d.append(train_d_loss)
    loss_history_g.append(train_g_loss)
    print("Epoch number {}\n Current loss {}\n".format(epoch, train_loss))

# Plot
plot_loss(loss_history, 'Loss')
plot_loss(loss_history_d, 'Training Discriminative Loss')
plot_loss(loss_history_g, 'Training Generative Loss')
plot_loss(train_acc, 'Accuracy')

positive = []
negative = []
tp = 0
tn = 0
fp = 0
fn = 0

threshold = 0.8

for i in range(len(test)):
    img0 = test_data[test[i][0]].unsqueeze(0)
    img1 = test_data[test[i][1]].unsqueeze(0)
    true_label = test[i][2]
    img0, img1 = img0.cuda(), img1.cuda()

    output_f1 = net(img0)
    output_f2 = net(img1)

    euclidean_distance = F.pairwise_distance(output_f1, output_f2)

    if true_label == 1:
        positive.append(euclidean_distance)
    else:
        negative.append(euclidean_distance)

    if euclidean_distance > threshold and true_label == -1:
        tn += 1
    if euclidean_distance > threshold and true_label == 1:
        fn += 1
    if euclidean_distance <= threshold and true_label == 1:
        tp += 1
    if euclidean_distance <= threshold and true_label == -1:
        fp += 1

print(tp)
print(fp)
print(tn)
print(fn)
print('Accuracy:', (tn + tp) / (tn + tp + fn + fp))
print('Recall: ', tp / (tp + fn))
print('Precision:', tp / (tp + fp))

# Visualize the results
plot_split(positive, negative)

# Classification test
test_classification(net, test_data, test_targets)
