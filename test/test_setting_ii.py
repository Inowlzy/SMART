import os
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from sklearn.manifold import TSNE
import seaborn as sns
from dataset import FeaturesDataset
from model_smart import EARNet
import matplotlib
font = {'family': 'sans-serif',
        'sans-serif': 'Helvetica'}
font = {'family': 'serif',
        'serif': 'Times New Roman',
        'weight': 'semibold'}
matplotlib.rc('font', **font)
color_map = sns.color_palette("cubehelix", 10)
classes = ['climb', 'climb walls', 'crouch', 'hand wave', 'hit', 'hit windows', 'run', 'sit', 'stand', 'walk']

def sample_points(class_position, target_num_points):
    num_points = class_position.size(0)
    class_position = class_position.cpu().numpy()
    if num_points <= target_num_points:
        num_new_points = target_num_points - num_points
        sampled_indices = np.random.choice(num_points, size=num_new_points, replace=True)
        sampled_point_cloud = np.vstack((class_position, class_position[sampled_indices]))
    else:
        step = num_points // target_num_points + 1
        sampled_point_cloud = class_position[::step]
        if sampled_point_cloud.shape[0] < target_num_points:
            num_new_points = target_num_points - sampled_point_cloud.shape[0]
            sampled_indices = np.random.choice(num_points, size=num_new_points, replace=True)
            sampled_point_cloud = np.vstack((sampled_point_cloud, class_position[sampled_indices]))
    return sampled_point_cloud

def object_separated_extraction(depth_tensor, mask_tensor):
    batch_size, _, sequence_length, height, width = depth_tensor.size()
    class_depth_batch = []
    class_position_batch = []
    for i in range(batch_size):
        class_depth_sequence = []
        class_position_sequence = []
        for j in range(sequence_length):
            depth = depth_tensor[i, :, j, :, :].view(-1)
            mask = mask_tensor[i, :, j, :, :].view(-1)
            mask1 = mask_tensor[i, 0, j, :, :]
            class_depths = []
            class_positions = []
            for class_idx in [0, 80, 159]:
                class_mask = (mask == class_idx)
                class_mask1 = (mask1 == class_idx)

                class_position = torch.nonzero(class_mask1, as_tuple=False)
                if class_idx != 80:
                    if class_position.size(0) > 0:
                        class_position = sample_points(class_position, 256)
                    else:
                        class_position = torch.zeros((256, 2)).cpu().numpy()
                    class_positions.append(class_position)

                good = torch.sum(class_mask) > 0
                if class_idx != 159:
                    if class_idx == 80 and not good:
                        class_mask = (mask == 0)
                        class_depth = torch.mean(depth[class_mask].float()).item()
                        class_depths.append(class_depth)
                    else:
                        class_depth = torch.mean(depth[class_mask].float()).item()
                        class_depths.append(class_depth)

            class_position_sequence.append(class_positions)
            class_depth_sequence.append(class_depths)
        class_position_batch.append(class_position_sequence)
        class_depth_batch.append(class_depth_sequence)

    class_depth_batch = torch.from_numpy(np.array(class_depth_batch)).unsqueeze(-1).permute(0, 2, 1, 3)
    class_position_batch = torch.from_numpy(np.array(class_position_batch)).permute(0, 2, 1, 3, 4)
    return class_depth_batch, class_position_batch

def test(model, test_loader, device):
    model.eval()
    all_labels = []
    all_predictions = []
    all_scene_predictions = []
    all_data = []

    with torch.no_grad(), tqdm(total=len(test_loader), desc="Test") as pbar:
        for action_tensor, depth_tensor, mask_tensor, label, scene_label in test_loader:
            class_depths, class_positions = object_separated_extraction(depth_tensor, mask_tensor)
            action_tensor, scene_tensor1, scene_tensor2, label, scene_label = (
                action_tensor.to(device),
                class_depths.to(device),
                class_positions.to(device),
                label.to(device),
                scene_label.to(device),
            )
            output, scene_weight, fusion_feature = model(action_tensor, scene_tensor1, scene_tensor2)
            _, predicted = torch.max(output, 1)

            all_data.extend(fusion_feature.cpu().numpy())
            all_labels.extend(label.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            all_scene_predictions.extend(scene_weight.cpu().numpy())

            pbar.update(1)

    confusion_mat = confusion_matrix(all_labels, all_predictions, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                                     normalize='true')
    confusion_mat_raw = confusion_matrix(all_labels, all_predictions, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    accuracy = np.sum(np.diag(confusion_mat_raw)) / np.sum(confusion_mat_raw)

    return confusion_mat,confusion_mat_raw, accuracy, all_scene_predictions, all_labels, all_data

def plot_confusion_matrix(confusion_mat, classes, epoch_num=None):
    plt.figure(figsize=(10, 8), dpi=100)
    plt.subplots_adjust(left=0.1, bottom=0.2)
    plt.tight_layout()
    plt.imshow(confusion_mat, interpolation="nearest", cmap=plt.cm.Blues)
    if epoch_num is None:
        plt.title("SMART")
        img_name = "SMART_test_setting_II.png"
    else:
        plt.title(f"Confusion Matrix of Training with Min Val Loss (SMART)")
        img_name = f"confusion_matrix_of_training_with_min_val_loss (SMART).png"
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, ha='right', fontsize=13)

    for i in range(len(classes)):
        for j in range(len(classes)):
            color = "white" if confusion_mat[i, j] > confusion_mat.max() / 2 else "black"
            plt.text(j, i, "{:.2f}".format(confusion_mat[i, j]), horizontalalignment="center", color=color, fontsize=10)

    plt.yticks(tick_marks, classes, fontsize=13)
    plt.ylabel("Truth label")
    plt.xlabel("Predict label")
    plt.savefig('../logf/confusion_matrix/' + img_name, bbox_inches='tight', dpi=400)
    plt.show()

def calculate_metrics(cm):
    num_classes = len(cm)
    total_samples = np.sum(cm)

    class_precision = []
    class_recall = []
    class_f1 = []

    for i in range(num_classes):
        true_positives = cm[i, i]
        false_positives = np.sum(cm[:, i]) - true_positives
        false_negatives = np.sum(cm[i, :]) - true_positives

        if true_positives + false_positives == 0:
            precision = 0.0
        else:
            precision = true_positives / (true_positives + false_positives)

        if true_positives + false_negatives == 0:
            recall = 0.0
        else:
            recall = true_positives / (true_positives + false_negatives)

        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)

        class_precision.append(precision)
        class_recall.append(recall)
        class_f1.append(f1)

    macro_precision = np.mean(class_precision)
    macro_recall = np.mean(class_recall)
    macro_f1 = np.mean(class_f1)

    ab_precision = np.mean([class_precision[0], class_precision[1], class_precision[4], class_precision[5]])
    ab_recall = np.mean([class_recall[0], class_recall[1], class_recall[4], class_recall[5]])
    ab_f1 = np.mean([class_f1[0], class_f1[1], class_f1[4], class_f1[5]])

    correct_predictions = np.sum(np.diag(cm))
    accuracy = correct_predictions / total_samples

    return class_precision, class_recall, class_f1, macro_precision, macro_recall, macro_f1, accuracy, ab_precision, ab_recall, ab_f1


def plot_embedding_2D(data, label):
    data = np.repeat(data, 4, axis=0)
    label = np.repeat(label, 4, axis=0)

    sorted_indices = np.argsort(label)
    label = label[sorted_indices]
    data = data[sorted_indices]

    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)
    fig = plt.figure(dpi=100)
    plt.tight_layout()
    plotted_labels = set()
    for i in range(data.shape[0]):
        if label[i] not in plotted_labels:
            plt.plot(data[i, 0], data[i, 1], marker='o', markersize=2, color=color_map[label[i]],
                     label=classes[label[i]], linestyle='None')
            plotted_labels.add(label[i])
        else:
            plt.plot(data[i, 0], data[i, 1], marker='o', markersize=2, color=color_map[label[i]])
    plt.xticks([])
    plt.yticks([])
    plt.legend(loc='best', fontsize=20, markerscale=2.5, labelspacing=0, handletextpad=0, borderpad=0, handlelength=0.8)
    return fig


# Main function
def main():
    # Define device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    gpus = [0]                      # the number must be the same as the number of GPUs in the device
    # Define hyperparameters
    scene_info = ['mask', 'depth']  # scene_info = [] or ['depth'] or ['mask'] or ['depth', 'mask']
    subjects = [5,6,7]              # subjects = [] or [1] or [1,2] or [1,2,3] or [1,2,3,4] ......
    dataset = FeaturesDataset(os.path.join(os.path.dirname(__file__), '..', 'Data'), scene_info=scene_info,
                              subjects=subjects)
    mini_batch = 16
    batch_size = mini_batch * len(gpus)
    shuffle_dataset = True
    random_seed = 42
    log_file = "../log/log_file/SMART_test_setting_II_log_" + str(datetime.now().strftime('%Y-%m-%d_%H-%M-%S')) + ".txt"
    # Creating data indices for training and validation splits:
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    test_indices = indices

    # Creating PT data samplers and loaders:
    test_sampler = SubsetRandomSampler(test_indices)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=test_sampler, drop_last=True)

    # Define model
    model = EARNet(num_actions=10,scene_info=scene_info,batch_size=mini_batch)  # 请替换为你自己的 EARNet 模型
    model.to(device)
    checkpoint = '../checkpoints/SMART.pth'
    model.load_state_dict(torch.load(checkpoint, map_location=device))

    # Define loss function and optimizer
    confusion_mat, confusion_mat_raw, test_accuracy, all_scene_predictions, all_labels, all_data = test(model, test_loader, device)
    
    # Visualize the t-SNE embedding and save the t-SNE embedding image
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    all_data = np.array(all_data)
    nan_list = ~np.isnan(all_data).any(axis=1)
    all_data = np.array(all_data[nan_list, :])
    all_labels = np.array(all_labels)
    all_labels = np.array(all_labels[nan_list])
    tsne_embedding = tsne.fit_transform(all_data)
    plot_embedding_2D(tsne_embedding, np.array(all_labels))
    plt.savefig('../log/tsne/SMART_test_setting_II_tsne.png', bbox_inches='tight', dpi=600)

    # Visualize the confusion matrix
    plot_confusion_matrix(confusion_mat, classes, epoch_num=None)
    np.save('../log/confusion_matrix/SMART_test_setting_II.npy', confusion_mat_raw)
    class_precision, class_recall, class_f1, macro_precision, macro_recall, macro_f1, accuracy, ab_precision, ab_recall, ab_f1 = calculate_metrics(
        confusion_mat_raw)
    formatted_precision = ["{:.4%}".format(precision) for precision in class_precision]
    formatted_recalls = ["{:.4%}".format(recall) for recall in class_recall]
    formatted_f1 = ["{:.4%}".format(f1) for f1 in class_f1]
    class_precision_dict = dict(zip(classes, formatted_precision))
    class_recalls_dict = dict(zip(classes, formatted_recalls))
    class_f1_dict = dict(zip(classes, formatted_f1))
    print("Class Precisions:", class_precision_dict)
    print("Class Recalls:", class_recalls_dict)
    print("Class F1 Scores:", class_f1_dict)
    print(f"Macro Precision: {macro_precision * 100:.4f}" + "%")
    print(f"Macro Recall: {macro_recall * 100:.4f}" + "%")
    print(f"Macro F1: {macro_f1 * 100:.4f}" + "%")
    print(f"Accuracy: {accuracy * 100:.4f}" + "%")
    print(f"Abnormal Precision: {ab_precision * 100:.4f}" + "%")
    print(f"Abnormal Recall: {ab_recall * 100:.4f}" + "%")
    print(f"Abnormal F1: {ab_f1 * 100:.4f}" + "%")
    print(f"Overall accuracy: {test_accuracy * 100:.4f}" + "%")
    with open(log_file, "a") as f:
        f.write(checkpoint + "\n")
        f.write("Class Precisions: " + "\n" + str(class_precision_dict) + "\n")
        f.write("Class Recalls: " + "\n" + str(class_recalls_dict) + "\n")
        f.write("Class F1 Scores: " + "\n" + str(class_f1_dict) + "\n")
        f.write(f"Macro Precision: {macro_precision * 100:.4f}" + '%' + "\n")
        f.write(f"Macro Recall: {macro_recall * 100:.4f}" + '%' + "\n")
        f.write(f"Macro F1: {macro_f1 * 100:.4f}" + '%' + "\n")
        f.write(f"Accuracy: {accuracy * 100:.4f}" + '%' + "\n")
        f.write(f"Abnormal Precision: {ab_precision * 100:.4f}" + '%' + "\n")
        f.write(f"Abnormal Recall: {ab_recall * 100:.4f}" + '%' + "\n")
        f.write(f"Abnormal F1: {ab_f1 * 100:.4f}" + '%' + "\n")
        f.write(f"Test Accuracy: {test_accuracy * 100:.4f}" + '%' + "\n")


if __name__ == "__main__":
    main()
