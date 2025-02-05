from torch.utils.data import DataLoader

from contribution_loss import JointClassifyCrossLoss, JointClassifyFocalLoss
from main_set import *
from models.apl_light import *
from utils import Format1, LRScheduler, EarlyStopping
from torchsummary import summary

batch_size = 64
num_epochs = 300

trainset = Format1('dataset', 'train')
testset = Format1('dataset', 'test')
num_train_samples = len(trainset)
num_test_samples = len(testset)
train_dataloader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(dataset=testset, batch_size=batch_size, shuffle=True)

# ----------------------------------------------------------------------------------------------------------------------
# main process
# ----------------------------------------------------------------------------------------------------------------------
model = ResNet(block=BasicBlock, layers=[1, 1, 1, 1], inchannel=52)
#model = ResNet(Bottleneck, [3, 4, 6, 3], use_se=False, use_cbam=True)
model = model.cuda()
summary(model, (52, 128))  # 假设输入的维度是 (52, 128)，请根据实际情况调整
# ----------------------------------------------------------------------------------------------------------------------
# 损失函数选择
# ----------------------------------------------------------------------------------------------------------------------
if args.loss_type == 'origin':  # 如果使用原始的loss
    criterion = nn.CrossEntropyLoss(reduction='sum').cuda()
elif args.loss_type == 'cross':  # 如果使用joint_cross_loss
    criterion = JointClassifyCrossLoss(weight=args.cross_weight, reduction='sum').cuda()
elif args.loss_type == 'focal':  # 如果使用joint_focal_loss
    criterion = JointClassifyFocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma, reduction='sum').cuda()

optimizer = torch.optim.RMSprop(model.parameters(), lr=3e-4)
scheduler = LRScheduler(optimizer)
early_stopping = EarlyStopping()

for epoch in range(num_epochs):
    print('Epoch:', epoch)
    # ------------------------------------------------------------------------------------------------------------------
    # 模型训练
    # ------------------------------------------------------------------------------------------------------------------
    model.train()
    correct_train_act = 0
    correct_train_loc = 0
    for samples, labels in train_dataloader:
        samples, labels = samples.to(args.device), labels.to(args.device)

        labels_act, labels_loc = labels[:, 0].squeeze(), labels[:, 1].squeeze()
        predict_label_act, predict_label_loc = model(samples)

        prediction = predict_label_loc.data.max(1)[1]
        correct_train_loc += prediction.eq(labels_loc.data.long()).sum()

        prediction = predict_label_act.data.max(1)[1]
        correct_train_act += prediction.eq(labels_act.data.long()).sum()

        optimizer.zero_grad()
        if args.loss_type == 'origin':  # 如果使用原始的loss
            loss_act = criterion(predict_label_act, labels_act)
            loss_loc = criterion(predict_label_loc, labels_loc)
            loss = loss_act + loss_loc
        elif args.loss_type == 'cross':  # 如果使用joint_cross_loss
            loss = criterion(predict_label_act, labels_act, predict_label_loc, labels_loc)
        elif args.loss_type == 'focal':  # 如果使用joint_focal_loss
            loss = criterion(predict_label_act, labels_act, predict_label_loc, labels_loc)

        loss.backward()
        optimizer.step()

    print("Activity Training accuracy:", (100 * float(correct_train_act) / num_train_samples))
    print("Location Training accuracy:", (100 * float(correct_train_loc) / num_train_samples))

    train_acc_act_str = str(100 * float(correct_train_act) / num_train_samples)[0:6]
    train_acc_loc_str = str(100 * float(correct_train_loc) / num_train_samples)[0:6]

    # ------------------------------------------------------------------------------------------------------------------
    # 模型测试集评估
    # ------------------------------------------------------------------------------------------------------------------
    model.eval()
    valid_loss = 0.0
    correct_test_act = 0
    correct_test_loc = 0
    for samples, labels in test_dataloader:
        with torch.no_grad():
            samples, labels = samples.to(args.device), labels.to(args.device)

            labels_act = labels[:, 0].squeeze()
            labels_loc = labels[:, 1].squeeze()
            predict_label_act, predict_label_loc = model(samples)

            prediction = predict_label_act.data.max(1)[1]
            correct_test_act += prediction.eq(labels_act.data.long()).sum()

            prediction = predict_label_loc.data.max(1)[1]
            correct_test_loc += prediction.eq(labels_loc.data.long()).sum()

            if args.loss_type == 'origin':  # 如果使用原始的loss
                loss_act = criterion(predict_label_act, labels_act)
                loss_loc = criterion(predict_label_loc, labels_loc)
                loss = loss_act + loss_loc
            elif args.loss_type == 'cross':  # 如果使用joint_cross_loss
                loss = criterion(predict_label_act, labels_act, predict_label_loc, labels_loc)
            elif args.loss_type == 'focal':  # 如果使用joint_focal_loss
                loss = criterion(predict_label_act, labels_act, predict_label_loc, labels_loc)

            valid_loss += loss.item()

    print("Activity Test accuracy:", (100 * float(correct_test_act) / num_test_samples))
    print("Location Test accuracy:", (100 * float(correct_test_loc) / num_test_samples))

    test_acc_act_str = str(100 * float(correct_test_act) / num_test_samples)[0:6]
    test_acc_loc_str = str(100 * float(correct_test_loc) / num_test_samples)[0:6]

    if epoch == 0:
        temp_test_acc = correct_test_act
        temp_train_acc = correct_train_act
    elif correct_test_act > temp_test_acc:
        torch.save(model,
                   f'checkpoints/{epoch}_'
                   f'train_acc_act：{train_acc_act_str[:6]}_loc：{train_acc_loc_str[:6]}_'
                   f'test_acc_act：{test_acc_act_str[:6]}_loc：{test_acc_loc_str[:6]}.pt')

        temp_test_acc = correct_test_act
        temp_train_acc = correct_train_act

    scheduler(valid_loss)
    #early_stopping(valid_loss)
    #if early_stopping.early_stop:
    #    break
