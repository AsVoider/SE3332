import re
import matplotlib.pyplot as plt

trainlosses = []
devloss = []
per = []
for i in range(1, 4):
    stri = "./p+j/log" + str(i) + ".out"
    with open(stri, 'r') as file:
        for line in file:
            # 使用正则表达式匹配包含"train_loss"的行
            match = re.search(r'train_loss = ([0-9.]+)', line)
            match2 = re.search(r'dev_loss = ([0-9.]+)', line)
            match3 = re.search(r'\'perplexity\': ([0-9.]+)', line)
            if match:
                # 提取train_loss的值并添加到数组中
                train_loss = float(match.group(1))
                trainlosses.append(train_loss)
            if match2:
                devl = float(match2.group(1))
                devloss.append(devl)
            if match3:
                pe = float(match3.group(1))
                per.append(pe)

plt.plot(trainlosses, label='Train Loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Training Loss Over Iterations')
plt.legend()
plt.show()

plt.plot(devloss, label='Dev Loss')
plt.xlabel('Iteration')
plt.ylabel('Dev Loss')
plt.title('Dev Loss Over Iterations')
plt.legend()
plt.show()

plt.plot(per, label='perplexity')
plt.xlabel('Iteration')
plt.ylabel('Perplexity')
plt.title('Perplexity Over Iterations')
plt.legend()
plt.show()
