import chainer
from chainer import cuda, Variable
import chainer.functions as F
import chainer.links as L
import numpy as np
from glimpseSensor import getGlimpses
from chainer import reporter


class RAM(chainer.Chain):
    def __init__(self, n_hidden = 256, n_units = 128, sigma = 0.03,
                 g_size=8, n_steps=6, n_depth=1, n_scale = 2, n_class = 2, n_in = 32, using_conv = False ):
        
        #n_in = g_size * g_size * 3
        #print(n_in)
        super(RAM, self).__init__(
            ll1=L.Linear(2, int(n_units)),           # 2 refers to x,y coordinate
            lc1=L.Convolution2D( 3, 32, 2),
            lc2=L.Convolution2D( 32, 32, 2),
            lc3=L.Convolution2D( 32, 32, 2),
            lc4=L.Convolution2D( 32, 32, 2),
            lrho1=L.Linear(n_in, int(n_units)),
            lh1=L.Linear(int(n_units)*2, n_hidden),
            lh2=L.Linear(n_hidden, n_hidden),
            lstm1=L.LSTM(n_hidden, n_hidden),
            lstm2=L.LSTM(n_hidden, n_hidden),
            ly=L.Linear(n_hidden, n_class),     # class/action output
            ll=L.Linear(n_hidden, 2),           # location output
            lb=L.Linear(n_hidden, 1),           # baseline output
        )
        self.g_size = g_size
        self.n_depth = n_depth
        self.n_scale = n_scale
        self.sigma = sigma
        self.n_steps = n_steps
        self.using_conv = using_conv

    def reset_state(self):
        self.lstm1.reset_state()
        self.lstm2.reset_state()

    def get_location_loss(self, l, mean):
        if chainer.config.train:
            term1 = 0.5 * (l - mean) ** 2 * self.sigma ** -2
            return F.sum(term1, axis=1).reshape(-1,1)
        else:
            xp = cuda.get_array_module(l)
            return Variable(xp.zeros(l.shape[0]))

    def sample_location(self, l):
        """
        sample new location from center l_data
        """
        if chainer.global_config.train:
            l_data = l.data
            bs = l_data.shape[0]
            xp = cuda.get_array_module(l_data)
            randomness = (xp.random.normal(0, 1, size=(bs, 2))).astype(np.float32)
            l_sampled = l_data + np.sqrt(self.sigma) * randomness
            return Variable(xp.array(l_sampled))
        else:
            return l

    def forward(self,n_in, x, l, first=False, ):
        if not first:
            centers = self.sample_location(l)
            ln_pi = self.get_location_loss(centers, l)


        else:
            centers = l
            ln_pi = self.get_location_loss(l, l) # ==0's
        rho = getGlimpses(x, centers, self.g_size, self.n_depth, self.n_scale, self.using_conv)
        print(rho.shape)
        g0 = F.relu(self.ll1(centers))
        print(g0.shape)
        c1 = F.relu(self.lc1(rho))
        print(c1.shape)
        c2 = F.relu(self.lc2(c1))
        print(c2.shape)
        mp1 = F.max_pooling_2d(c2,2)
        print(mp1.shape)
        c3 = F.relu(self.lc3(mp1))
        print(c3.shape)
        c4 = F.relu(self.lc4(c3))
        print(c4.shape)
        mp2 = F.max_pooling_2d(c4,2)
        print(mp1.shape)
        flat1 = F.flatten(mp2)
        print(flat1.shape)
        
        g1 = F.relu(self.lrho1(flat1.reshape(n_in,-1)))
        print(g1.shape)
        h0 = F.concat([g0, g1], axis=1)
        print(h0.shape)
        h1 = F.relu(self.lh1(h0))
        print(h1.shape)
        h2 = F.relu(self.lh2(h1))
        print(h2.shape)
        h_out1 = self.lstm1(h2)
        print(h_out1.shape)
        h_out2 = self.lstm2(h_out1)
        print(h_out2.shape)
        y = self.ly(h_out2)
        l_out = F.tanh(self.ll(h_out2))
        b = F.sigmoid(self.lb(h_out2))
        return l_out, ln_pi, y, b



    def __call__(self, x, t):

        x = chainer.Variable(self.xp.asarray(x))
        t = chainer.Variable(self.xp.asarray(t))
        #print(x.shape)
        #print(t.shape)
        batchsize = x.data.shape[0]
        self.reset_state()

        # initial l
        l = np.random.uniform(-1, 1, size=(batchsize, 2)).astype(np.float32)
        l = chainer.Variable(self.xp.asarray(l))

        sum_ln_pi = Variable((self.xp.zeros((batchsize,1))))
        sum_ln_pi = F.cast(sum_ln_pi,'float32')
        l, ln_pi, y, b = self.forward(l.shape[0],x, l, first=True, )
        for i in range(1,self.n_steps):
            l, ln_pi, y, b = self.forward(l.shape[0],x, l)
            ln_pi = F.cast(ln_pi,'float32')
            sum_ln_pi += ln_pi
        y = y + 0.00000001
        self.loss_action = F.softmax_cross_entropy(y, t)
        self.loss = self.loss_action
        self.accuracy = F.accuracy(y, t)
        reporter.report({'accuracy': self.accuracy}, self)
        reporter.report({'cross_entropy_loss': self.loss}, self)
        #print(y.shape)
        self.y = F.argmax(y, axis=1)
        #print(self.y)
        #self.t = F.argmax(t, axis=1)
        #reporter.report({'actual': t}, self)
        #reporter.report({'predicted': self.y}, self)
        if chainer.global_config.train:

            conditions = self.xp.argmax(y.data, axis=1) == t.data
            r = self.xp.where(conditions, 1., 0.).astype(self.xp.float32)
            r = self.xp.expand_dims(r, 1)
            # squared error between reward and baseline
            self.loss_baseline = F.mean_squared_error(r, b)
            self.loss += self.loss_baseline
            # loss with reinforce rule
            mean_ln_pi = sum_ln_pi / (self.n_steps - 1)
            a = F.sum(-mean_ln_pi * (r - b)) / batchsize
            self.reinforce_loss = F.sum(-mean_ln_pi * (r-b)) / batchsize
            self.loss += self.reinforce_loss
            reporter.report({'cross_entropy_loss': self.loss_action}, self)
            #reporter.report({'reinforce_loss': self.reinforce_loss}, self)
            #reporter.report({'total_loss': self.loss}, self)
            reporter.report({'train_accuracy': self.accuracy}, self)

        #print(self.loss)
        return self.loss


if __name__ == "__main__":
    train, test = chainer.datasets.get_mnist(withlabel=True, ndim=3)
    train_data, train_targets = np.array(train).transpose()
    train_data = np.array(list(train_data)).reshape(train_data.shape[0], 3, 28, 28)
    train_targets = np.array(train_targets).astype(np.int32)
    x = train_data[0:2]
    t = train_targets[0:2]
    model = RAM()
    model.to_gpu()
    model(x, t)
