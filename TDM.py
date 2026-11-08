import pandas as pd 
import torch 
import torch.nn as nn  
import torch.optim as optim
import sklearn  

#PANDAS WORKING HERE----////
df=pd.read_csv("Titanic_Cleaned.csv")

#BRINGING IN THE train_test_split TO SEPARATE INPUTS FEATURES AND ANSWERS
from sklearn.model_selection import train_test_split

X=df.drop(columns=["survived"])
y=df["survived"]

X_train, X_test, y_train, y_test= train_test_split(X,y, test_size=0.2, random_state=42)

#convert the training data to tensors
X_train=torch.tensor(X_train.values.astype(float), dtype=torch.float32)
y_train=torch.tensor(y_train.values.astype(float), dtype=torch.float32).view(-1, 1)

#convert the testing data to tensors
X_test=torch.tensor(X_test.values.astype(float), dtype=torch.float32)
y_test=torch.tensor(y_test.values.astype(float), dtype=torch.float32).view(-1, 1)

#using standard scaler to reduce the age and fare numbers
from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()

df[["age", "fare"]]=scaler.fit_transform(df[["age", "fare"]])
print("Scaled:", df.head)


#USING DATA LOADER TO FEED MODEL IN BATCHES
from torch.utils.data import TensorDataset, DataLoader

dataset =TensorDataset(X_train,y_train,)
dataset =TensorDataset(X_test, y_test)
loader=DataLoader(dataset, batch_size=32, shuffle=True)

for batch_X, batch_y in loader:
	print("Batch X:", batch_X)
	print("Batch y:", batch_y)
	print("---")
# TRAINING LOOP////////////

my_layers=nn.Sequential(
	nn.Linear(8,32),
	nn.ReLU(),
	nn.Linear(32,16),
	nn.ReLU(),
	nn.Linear(16, 1),
	nn.Sigmoid()
	)
#DECLARING THE LOSS FUNCTION AND THE OPTIMIZER
loss_function=nn.BCELoss()
optimizer=optim.Adam(my_layers.parameters(), lr=0.005)

# THE TRAINING LOOP
for epoch in range(2000):
	for batch_X, batch_y in loader: 
		optimizer.zero_grad()
	prediction=my_layers(X_train)
	loss=loss_function(prediction, y_train)
	loss.backward()
	optimizer.step()

print("Final prediction:", prediction)
print("Final Loss:", loss.item())


#turn off gradient tracking for testing (saves memory)
with torch.no_grad():
	test_predictions= my_layers(X_test)

#we round the predictions to 0 or 1
predicted_classes=(test_predictions >0.5).float()

#compare to the actual answers
correct = (predicted_classes ==y_test).sum().item()
total = y_test.size(0)
accuracy = correct / total

print(f"Test Accuracy: {accuracy * 100:.2f}%")




