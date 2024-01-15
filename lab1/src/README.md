### LR
1. load the data and turn it to numpy form
2. defined function:
   1. loss(z, y) -- loss function in LR
      1. z is pred 
      2. y is label
   2. grad(x, pre, y) -- gradient function (maybe unused)
      1. x -- input
      2. pre -- res in process
      3. y -- label
   3. gradientdescent(x, y) -- training function
      1. x -- input
      2. y -- label
      3. return w, loss
3. if pred > 5 then pred = 1 else pred = 0; then count the number of (pred == label)

### SVM
1. load the data
2. class SVM: init(self, learning_rate, lambda_param, n_iters) -- easy to understand
   1. fit(self, x, y) -- train
      1. x -- input
      2. y -- label
   2. updata(self, x, y) -- update one W
      1. x -- one of input
      2. y -- corresponding label
      3. return loss
   3. predect(self, X)
      1. X -- test input
      2. return pred_label
3. then use SVM class to train and test

### MLP
1. load and transform the data
2. class CustromDataset(Dataset):
   1. init(self, data, labels) -- chenge data to the format of torch
   2. len(self) -- return data.length
   3. getitem(self, index) -- use index to get the data
3. class MLP(torch.nn.Mouble):
   1. init(self, input_dim, hidden_dim, output_dim) define layers
   2. forward(self, x) x -> layer1 -> activation -> layer2 -> activation -> output
      1. x input
      2. return forward x
4. def training(model): use model to train data
   
5. def tes_ting(model): just test
6. use pre-defined class / function to train & test

### it uses relative path to load data, so I have to up load the dataset;