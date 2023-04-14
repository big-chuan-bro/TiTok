import torch
from sklearn import metrics
from torch.autograd import Variable
import torch.nn as nn


def evaluate_from_dataloader(model_instance, input_loader, monte_carlo=False):
    with torch.no_grad():
        if monte_carlo is False:
            return evaluate_from_dataloader_basic(model_instance, input_loader)
        else:
            return evaluate_from_dataloader_monte_carlo(model_instance, input_loader)


def evaluate_from_dataloader_monte_carlo(model_instance, input_loader):
    
    all_probs = []
    for sample_i in range(model_instance.args.monte_carlo_sample_size):
        model_stats, probs_i = evaluate_from_dataloader_basic(model_instance, input_loader)
        all_probs.append(probs_i)
    probs = torch.stack(all_probs)
    probs_avg = probs.mean(dim=0)
    return model_stats, probs_avg


def evaluate_from_dataloader_basic(model_instance, input_loader):
    
    num_iter = len(input_loader)
    iter_test = iter(input_loader)
    first_test = True

    for i in range(num_iter):
        model_instance.eval()
        data = iter_test.next()
        indices = data[0]
        inputs = data[1]
        labels = data[2]
        


        if model_instance.use_gpu:
            inputs = Variable(inputs.cuda())
            labels = Variable(labels.cuda())
            
        else:
            inputs = Variable(inputs)
            labels = Variable(labels)
  

        probabilities, t_output = model_instance(inputs)
        probabilities = nn.Softmax(dim=1)(probabilities)
        probabilities = probabilities.data.float()
        labels = labels.data.float()
        
        if first_test:
            all_probs = probabilities
            all_labels = labels
            first_test = False
        else:
            all_probs = torch.cat((all_probs, probabilities), 0)
            all_labels = torch.cat((all_labels, labels), 0)
        model_instance.train()    

    predictions = all_probs.argmax(dim=1)
    accuracy = torch.sum(predictions == all_labels).float() / float(all_labels.size()[0])

    avg_acc = metrics.balanced_accuracy_score(all_labels.cpu().numpy(), predictions.cpu().numpy())

    
    model_stats = {'accuracy': accuracy.item(),
                   'test_balanced_acc': avg_acc}
    return model_stats, all_probs


def evaluate_from_batch(model_instance, input_x, input_y):
    with torch.no_grad():
       
        probabilities = model_instance.predict(input_x)
        probabilities = probabilities.data.float()
        labels = input_y.data.float()
        _, predict = torch.max(probabilities, 1)
        accuracy = torch.sum(torch.squeeze(predict).float() == labels).float() / float(labels.size()[0])
        
        return {'accuracy': accuracy.item()}
