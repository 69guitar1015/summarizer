# Summarizer   

The wrapper of TensorBoardX
You can write inference functions with summarization codes
- summarization will work only in `with summarizer.enable():`
- no need to write redundant `if - else`
- no need to pass a `SummaryWriter` instance to subnetworks
  - summarizer have a `Summarizer` instance as module variable
  - all you need is `import summarizer` in each source codes

## Usage
```python
import chainer
import summarizer
summarizer.initialize_writer(logdir='results')

def MLP(chainer.Chain):
    def __init__(self):
    # no need to write this
    # def __init__(self, writer):
    #     self.writer = writer
        super(MLP, self).__init__()
        with self.init_scope():
            self.l1 = chainer.links.Linear(100)
            self.l2 = chainer.links.Linear(1)
    def __call_(x):
        h = chainer.functions.relu(self.l1(x))
        h = chainer.functions.sigmoid(self.l2(x))
        
        # no need to write this
        # if something.debug:
        #     self.writer.add_histogram('l1_W', self.l1.W)
        
        summarizer.add_histogram('l1_W', self.l1.W) # these methods works only in summarizer.enable()
        summarizer.add_histogram('l1_b', self.l1.b)
        summarizer.add_histogram('l2_W', self.l2.W)
        summarizer.add_histogram('l2_b', self.l2.b)
        
        
        return h

mlp = MLP()

# writer is disable
loss = mlp(x_train)

# writer is enable
with summarizer.enable():
    loss = mlp(x_validation)
```
