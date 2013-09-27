import sys
import os
import logging
import numpy
import random
import math
import pickle
from itertools import groupby
import operator

LOGGER_NAME = "logger"
DELTA = 0.001

class HMM:
    T = 0

    N = 0
    M = 0
    A = []
    B = []
    pi = []
    
    alpha = []
    beta = []
    gamma = []
    xi = []

    seq_map = {}
    O = []
    O_map = []

    scale = []
    prob = 0.0

    def __init__(self):
        #set_logger()
        log = logging.getLogger(LOGGER_NAME)
        log.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')

        handler_stream = logging.StreamHandler()
        handler_stream.setFormatter(formatter)
        log.addHandler(handler_stream)
    
        handler_file = logging.FileHandler('errors.txt')
        handler_file.setFormatter(formatter)
        handler_file.setLevel(logging.INFO)
        log.addHandler(handler_file)
        self.logger = log
    
    def ProcessInput(self, name1, name2):
        self.logger.debug(name1 + " " + name2)

        #Read the input files
        [pid_file1, seq_file1] = zip(*[line.split(' ') for line in open(name1)])
        [pid_file2, seq_file2] = zip(*[line.split(' ') for line in open(name2)])
        
        #pid and seq contain the combined lists
        pid = list(pid_file1) + list(pid_file2)
        seq = list(seq_file1) + list(seq_file2)
        #Remove newline characters from seq
        seq = map(lambda s: s.strip(), seq)
        #self.logger.debug(pid)
        #self.logger.debug(seq)
        
        self.O = seq
        self.ProcessSequence()

    def ProcessSequence(self, baseline=False):
        seq = self.O
        #uniquify seq, maintain sequence
        seen = set()
        seq_unique = [x for x in seq if x not in seen and not seen.add(x)]
        #self.logger.debug(seq)

        #declare N and M
        if not baseline:
            self.N = len(seq_unique)
            self.M = self.N
            self.logger.debug(self.N)
        else:
            self.N = 22
            self.M = 22

        #Create a index of the states
        di = dict(zip(seq_unique, xrange(len(seq_unique))))
        self.seq_map = di

        #Map the original sequence to the indexed states
        seq_mapped = [di[s] for s in seq]
        #self.logger.debug(seq_mapped)

        self.O_map = seq_mapped
        self.T = len(seq_mapped)

    #Generate random matrix for A, B and pi
    def GenerateRandomMatrix(self):
        pi_sum = 0

        for x in range(0, self.N):
            a = []
            b = []
            a_row_sum = 0
            b_row_sum = 0

            for x in range(0, self.N):
                a_r = random.randrange(0,10)
                a.append(float(a_r))
                a_row_sum = a_row_sum + a_r

                b_r = random.randrange(0, 10)
                b.append(float(b_r))
                b_row_sum = b_row_sum + b_r

            for x in range(0, len(a)):
                a[x] = a[x]/a_row_sum
                b[x] = b[x]/b_row_sum
            
            self.A.append(a)
            self.B.append(b)

            pi_r = random.randrange(0, 10)
            self.pi.append(float(pi_r))
            pi_sum = pi_sum + pi_r

        for x in range(0, self.N):
            self.pi[x] = self.pi[x]/pi_sum

        #self.logger.debug(self.A)
        #self.logger.debug(self.B)
        #self.logger.debug(self.pi)
    
    def InitializeVars(self):
        x = []
        for k in range(self.T):
            xxi = [[0.0 for i in range(self.N)] for j in range(self.N)]
            x.append(xxi)

        self.xi = x
        self.gamma = [[0.0 for i in range(self.N)] for j in range(self.T)]
        
        self.scale = [0.0 for x in range(self.T)]
        self.alpha = [[0.0 for x in range(self.N)] for i in range(self.T)]
        self.beta = [[0.0 for i in range(self.N)] for j in range(self.T)]

    def BaumWelch(self):

        deltaprev = 10e-70
        l = 0
        
        self.InitializeVars()

        logprobf = self.Forward()
        logprobinit = logprobf
        self.Backward()
        self.ComputeGamma()
        self.ComputeXi()
        logprobprev = logprobinit

        while True:
            #Reestimate frequency of state i in time t = 0
            for i in range(0, self.N):
                self.pi[i] = 0.1 + 0.9*self.gamma[0][i]
            
            #Reestimate transition matrix and symbol prob in each state
            for i in range(0, self.N):
                denominatorA = 0.0
                for t in range(0, self.T - 1):
                    denominatorA += self.gamma[t][i]

                for j in range(0, self.N):
                    numeratorA = 0.0
                    for t in range(0, self.T - 1):
                        numeratorA += self.xi[t][i][j]
                    self.A[i][j] = 0.1 + 0.9 * numeratorA/denominatorA
                    
                denominatorB = denominatorA + self.gamma[self.T - 1][i]
                
                for k in range(0, self.M):
                    numeratorB = 0.0
                    for t in range(0, self.T):
                        if (self.O_map[t] == k):
                            numeratorB += self.gamma[t][i]
                    
                    self.B[i][k] = 0.1 + 0.9 * numeratorB/denominatorB
            
            logprobf = self.Forward()
            self.Backward()
            self.ComputeGamma()
            self.ComputeXi()
            
            delta = logprobf - logprobprev
            self.logger.debug("%d %f" % (logprobf, delta))
            logprobprev = logprobf
            l += 1
            self.logger.debug(l)
            
            if (delta < DELTA):
                break;
        self.logger.debug(delta)
        return [l, logprobinit, logprobf]

    def ComputeGamma(self):
        for t in range(0, self.T):
            denominator = 0.0
            for j in range(0, self.N):
                self.gamma[t][j] = self.alpha[t][j] * self.beta[t][j]
                denominator += self.gamma[t][j]

            #self.logger.debug(self.gamma[t])
            for i in range(0, self.N):
                self.gamma[t][i] = self.gamma[t][i]/denominator
                
    def ComputeXi(self):
        for t in range(0, self.T - 1):
            s = 0.0
            for i in range(0, self.N):
                for j in range(0, self.N):
                    self.xi[t][i][j] = self.alpha[t][i] * self.beta[t+1][j] * self.A[i][j] * self.B[j][self.O_map[t+1]]
                    s += self.xi[t][i][j]
            
            for i in range(0, self.N):
                for j in range(0, self.N):
                    self.xi[t][i][j] /= s

    def Forward(self):

        #Initialization
        self.scale[0] = 0.0
        for x in range(0, self.N):
            self.alpha[0][x] = self.pi[x] * self.B[x][self.O_map[0]]
            self.scale[0] += self.alpha[0][x]

        for x in range(0, self.N):
            self.alpha[0][x] /= self.scale[0]

        #Induction
        for t in range(0, self.T -1):
            self.scale[t+1] = 0.0
            for j in range(0, self.N):
                s = 0.0
                for i in range(0, self.N):
                    s += self.alpha[t][i] * self.A[i][j]

                self.alpha[t+1][j] = s * self.B[j][self.O_map[t+1]]
                self.scale[t+1] += self.alpha[t+1][j]
            
            for j in range(0, self.N):
                self.alpha[t+1][j] /= self.scale[t+1]

        self.prob = 0.0
        
        #Termination
        for t in range(0, self.T):
            self.prob += math.log(self.scale[t])

        return self.prob
    
    def Backward(self):

        #Initialization
        for i in range(0, self.N):
            self.beta[self.T-1][i] = 1.0/self.scale[self.T-1]

        #Induction
        for t in xrange(self.T - 2, -1, -1):
            for i in xrange(0, self.N):
                s = 0.0
                for j in xrange(0, self.N):
                    s += self.A[i][j] * (self.B[j][self.O_map[t+1]]) * self.beta[t+1][j]
                self.beta[t][i] = s/self.scale[t]
        

    def Start(self, name1, name2):
        self.ProcessInput(name1, name2)
        self.GenerateRandomMatrix()
        [no_iter, probinit, probfinal] = self.BaumWelch()
        self.logger.info("Number of iterations - %d" % no_iter)
        self.logger.info("Log Prob(observation | init model) - %d" % probinit)
        self.logger.info("Log Prob(observation | estimated model) - %d" % probfinal)
        pickle.dump(self.A, open("a.lambda", "wb"))
        pickle.dump(self.B, open("b.lambda", "wb"))
        pickle.dump(self.pi, open("pi.lambda", "wb"))
    
    def Test(self, name1, name2, test):
        self.Unpickle()
        baseline = {}
        #self.InitializeVars()
        self.EstablishBaseline([name1, name2], baseline)
        testdata = {}
        self.EstablishBaseline([test], testdata)
        self.logger.debug(baseline)
        self.logger.debug(testdata)
        #self.TestData(test)

    def EstablishBaseline(self, files, baseline):
        #files = [name1, name2]
        #baseline = {}
        for f in files:
            di = self.LoadData(f)
            for key in di.keys():
                self.O = di[key]
                self.ProcessSequence(True)
                #[n, i, f] = self.BaumWelch()
                self.InitializeVars()
                f = self.Forward()
                baseline[key] = f
        self.sorted_baseline = sorted(baseline.iteritems(), key = operator.itemgetter(1))
        #self.logger.debug(self.sorted_baseline)

    def Unpickle(self):
        self.A = pickle.load(open("a.lambda", "rb"))
        self.B = pickle.load(open("b.lambda", "rb"))
        self.pi = pickle.load(open("pi.lambda", "rb"))

        self.logger.debug(self.A)
        self.logger.debug(self.B)
        self.logger.debug(self.pi)

    def LoadData(self, name):
        things = [map(lambda s:s.strip(), line.split(' ')) for line in open(name)]
        di = {}
        for key, group in groupby(things, lambda x: x[0]):
            li = []
            for thing in group:
                li.append(thing[1])
            di[key] = li
        return di

def main():
    hmm = HMM()
    hmm.Start(sys.argv[1], sys.argv[2])
    hmm.Test(sys.argv[1], sys.argv[2], sys.argv[3])

if __name__ == "__main__":
    sys.exit(main())
                                                  
        
