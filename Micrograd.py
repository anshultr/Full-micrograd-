from importlib.metadata import requires
import numpy as np
import matplotlib.pyplot as plt
import math
 
# ================================================================
# PART 0 -- numerical derivative warm-up (unchanged, was already correct)
# ================================================================
h = 0.00001
a = 2.0
b = -3.0
c = 10.0
d1 = a * b + c
a += h
d2 = a * b + c
print("d1 =", d1)                              # 4.0
print("d2 =", d2)                               # 3.99997  (a nudged by h)
print("numerical d(a*b+c)/da =", (d2 - d1) / h) # -3.0  == b, matches analytically
 
 
# ================================================================
# PART 1 -- the Value class (the autograd engine)
# ================================================================
class Value:
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._prev_ordered = _children  # order-preserving copy, used only by the graph visualizer in PART 5
        self._op = _op
        self.label = label
 
    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"
 
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other) 
        out = Value(self.data + other.data, (self, other), '+')
        def _backward():
            # += will acumulate the gradients on top of each other not overwriting them
            self.grad += 1.0 * out.grad 
            other.grad += 1.0 * out.grad 

        out._backward = _backward
        return out
 
    def __mul__(self, other):
        # self = self if isinstance(Value, self) else Value(self) 
        other = other if isinstance(other, Value) else Value(other) # we will initiate the random value into the other 
        out = Value(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad += other.data * out.grad   # BUG FIX: += not =
            other.grad += self.data * out.grad    # BUG FIX: += not =
        out._backward = _backward
        return out


    def __pow__(self, other): # other will be the power of self and other = into or float only 
        assert isinstance(other, (int, float)) 
        out = Value(self.data ** other, (self, ), f'**{other}') 
        def _backward(): 
            # backward for the power rule is d/dx(x^n) = n*x^ n - 1 and here n = other and x = self 
            self.grad += other * (self.data ** (other - 1)) * out.grad 
        out._backward = _backward 
        return out 
    def __rmul__(self, other): 
        return self * other 
 # a = 12, b =?? a / b = a * (b ** -1) a = self and b =other.data 
    def __truediv__(self, other):
        return self * other ** -1 
    def __sub__(self): 
        return self * -1 
    def __sub__(self, other): # self - other 
        return self + other * -1
 
    def tanh(self):
        n = self.data
        t = (math.exp(2 * n) - 1) / (math.exp(2 * n) + 1)
        
        out = Value(t, (self,), 'tanh')
        def _backward():
            self.grad += (1 - t**2) * out.grad   # BUG FIX: += not =
        out._backward = _backward
        return out
 
    def sigmoid(self):
        n = self.data
        s = 1 / (1 + math.exp(-n))
        out = Value(s, (self,), 'sigmoid')
        def _backward():
            self.grad += s * (1 - s) * out.grad
     
        out._backward = _backward
        return out
    def exp(self): 
        x = self.data 
        out = Value(math.exp(x), (self, ), 'exp') 
        def _backward():
            self.grad += out.data * out.grad 
        out._backward = _backward 
        return out 

    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
            
        build_topo(self)
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

 
# ================================================================
# PART 2 -- simple expression graph:  L = (a*b + c) * f
# ================================================================
a = Value(2.0, label='a')
a + 15 
print(a) 
b = Value(-3.0, label='b')
c = Value(10.0, label='c')
e = a * b;  e.label = 'e'
d = e + c;  d.label = 'd'
f = Value(-2.0, label='f')
L = d * f;  L.label = 'L'
print(f"\nforward pass: e={e.data}, d={d.data}, L={L.data}")   # e=-6, d=4, L=-8
# ---- manual (hand-derived) gradients, as a sanity check ----
a.grad = -2.00 * -3.00   #  dL/da = dL/de * de/da = f * b =  6.0
b.grad = -2.00 * 2.00    #  dL/db = dL/de * de/db = f * a = -4.0
print(f"hand-derived  a.grad={a.grad}  b.grad={b.grad}")
 
# ---- automatic gradients via L.backward(), called ONCE on the true output ----

a.grad = 0.0
b.grad = 0.0
L.backward()
print(f"L.backward()  a.grad={a.grad}  b.grad={b.grad}  c.grad={c.grad}  f.grad={f.grad}")
# a.grad=6.0 b.grad=-4.0 (matches the hand-derived values) c.grad=-2.0 f.grad=4.0 (new)
 
# ---- isolated numerical check of dL/da  ----

h = 0.0001
a_bumped = Value(a.data + h)
e_check = a_bumped * b
d_check = e_check + c
L_check = d_check * f
numerical_dLda = (L_check.data - L.data) / h
print(f"numerical dL/da = {numerical_dLda}   (compare to a.grad = {a.grad})")
 
# ---- one step of gradient descent on a, b, c, f ----
a.data += 0.01 * a.grad
b.data += 0.01 * b.grad
c.data += 0.01 * c.grad
f.data += 0.01 * f.grad
e = a * b
d = e + c
L_new = d * f
print(f"L after one gradient-descent step: {L_new.data}   (was {L.data}; should move toward 0)")
 
 
# ================================================================
# PART 4 -- a single neuron: forward + backward
# ================================================================
x1 = Value(2.0, label='x1')
x2 = Value(0.0, label='x2')
w1 = Value(-3.0, label='w1')
w2 = Value(1.0, label='w2')
bias = Value(6.8, label='b')

x1w1 = x1 * w1; x1w1.label = 'x1w1'
x2w2 = x2 * w2; x2w2.label = 'x2w2'
x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label = 'x1w1x2w2'
n = x1w1x2w2 + bias; n.label = 'n'
o = n.tanh(); o.label = 'o'
print(f"\nn = {n.data}")        # 0.8
print(f"o = tanh(n) = {o.data}")  # ~0.6640, now correctly bounded in (-1,1)
 
# ---- manual backward, using the REAL local derivative ----
o.grad = 1.0
df = 1 - o.data ** 2   # do/dn, now a genuine number instead of a mismatched 0.5
print(f"do/dn = {df}")
 
n.grad = df
x1w1x2w2.grad = df
bias.grad = df
x1w1.grad = df
x2w2.grad = df
 
x1.grad = x1w1.grad * w1.data
w1.grad = x1w1.grad * x1.data
x2.grad = x2w2.grad * w2.data
w2.grad = x2w2.grad * x2.data
print(f"manual  x1.grad={x1.grad}  w1.grad={w1.grad}  x2.grad={x2.grad}  w2.grad={w2.grad}")
 
x1b = Value(2.0, label='x1'); x2b = Value(0.0, label='x2')
w1b = Value(-3.0, label='w1'); w2b = Value(1.0, label='w2')
biasb = Value(6.8, label='b')
x1w1_b = x1b * w1b; x2w2_b = x2b * w2b
x1w1x2w2_b = x1w1_b + x2w2_b
n_b = x1w1x2w2_b + biasb
# o_b = n_b.tanh()
# o_b.backward()   # single call, propagates through the whole graph automatically
e = (2*n).exp() 
o = (e - 1)/ (e + 1) 
o.label = 'o' 
o._backward()
print(f"printing o here {o}") 
print(f" printing x1 here {x1}") 

print(f"auto    x1.grad={x1b.grad}  w1.grad={w1b.grad}  x2.grad={x2b.grad}  w2.grad={w2b.grad}")
print(f"auto    n.grad={n_b.grad}  b.grad={biasb.grad}  x1w1x2w2.grad={x1w1x2w2_b.grad}")
# we are having an error a.grad should not be 1.0, we deposit b = a.grad two times two a 
# Calculates gradient from the first a: a.grad = 1.0 then Calculates gradient from the second a: a.grad = 1.0 (overwriting the first one) because we dont have += sign
a = Value(3.0, label = 'a')
b = a + a; b.label = 'b' 
b.backward() 
print(f"printing the b {b}")
a.backward() 
print(f" printing the a value ({a})")
# the pytorch segment 
import torch 
                                      # here torch does not allow to backprpogate we cannot directly print x1.grad we have to take permission first 
x1 = torch.tensor([2.0]).double()     ;x1.requires_grad = True 
x2 = torch.tensor([0.0]).double()     ;x2.requires_grad = True 
w1 = torch.tensor([-3.0]).double()    ;w1.requires_grad = True 
w2 = torch.tensor([1.0]).double()     ;w2.requires_grad = True 
b = torch.tensor([6.88137]).double()  ;b.requires_grad = True 

n = w1*x1 + w2*x2 + b 
o = torch.tanh(n) 
print(o.data.item())
o.backward() 

print('x1', x1.grad.item())
print('x2', x2.grad.item()) 
print('w1', w1.grad.item()) 
print('w2', w2.grad.item()) 
import random 
class Neuron: 
    def __init__(self, nin): # here nin = number of inputs coming to a neuron it would be np.sum( w * x + b) 
        self.w = [Value(random.uniform(-1, 1)) for _ in range (nin)] 
        self.b = Value(random.uniform(-1, 1))
    def __call__(self, x): 
        # n = w * x + b we will zip this where w * x will create new tuples 
        # forward pass 
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        # also add non linearity 
        out = act.tanh() 
        return out 
    def parameters(self): 
        return self.w + [self.b]
class layer: 
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)] 
    def __call__(self, x):          
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs
    def parameters(self): 
        params = [] 
        for neuron in self.neurons: 
            ps = neuron.parameters()
            params.extend(ps)
        return params 

class MLP: 
    def __init__(self, nin, nout): 
        sz = [nin] + nout   # here sz = outcome 
        self.layers = [layer(sz[i], sz[i + 1]) for i in range(len(nout))] 
    def __call__(self, x): 
        for layer in self.layers: 
            x = layer(x) 
        return x 
    def parameters(self): 
        params = [] 
        for layer in self.layers:
            ps = layer.parameters()
            params.extend(ps) 
        return params 
# here xy is the output that we have predicted 
n = MLP(3, [4, 4, 1]) 
xs = [
    [2.0, 3.0, 4.0], 
    [1.2, -4.5, 5.0], 
    [1.4, -3.0, 2.0],
    [2.0, 3.2, 0.5]
]
ys = [1.0, -1.0, -1.0, 1.0] 
for k in range(20): 
    y_pred = [n(x) for x in xs]
    loss = sum(((yout - ygt)**2 for yout, ygt in zip(y_pred, ys)), start=Value(0.0))

    # backward pass 
    for p in n.parameters(): 
        p.grad = 0.0 
    loss.backward()
    for p in n.parameters(): 
        p.data += -0.05 * p.grad 
    print(f" --- PRINTING THE LOSS OF THE DATA {k, loss.data}") 
