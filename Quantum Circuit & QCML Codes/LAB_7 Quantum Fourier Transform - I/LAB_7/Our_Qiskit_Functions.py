from qiskit import ClassicalRegister, QuantumRegister, QuantumCircuit, Aer, execute
from qiskit.extensions.simulator import snapshot
from qiskit.tools.visualization import circuit_drawer
import numpy as np
import math as m
import scipy as sci
import random
import time
import matplotlib
import matplotlib.pyplot as plt

S_simulator = Aer.get_backend('statevector_simulator')
M_simulator = Aer.get_backend('qasm_simulator')

def Wavefunction(obj, **kwargs):
    '''
    Prints a tidier version of the array statevector corresponding to the wavefunction of a QuantumCircuit object
    Keyword Arguments:          precision (integer) - the decimal precision for amplitudes
                                column (Bool) - prints each state in a vertical column
                                systems (array of integers) - separates the qubits into different states
                                show_systems (array of Bools) - indicates which qubit systems to print
    '''
    if type(obj) == QuantumCircuit:
        statevec = execute(obj, S_simulator, shots=1).result().get_statevector()
    if type(obj) == np.ndarray:
        statevec = obj
    
    sys = False
    NL = False
    dec = 5
    
    if 'precision' in kwargs:
        dec = int(kwargs['precision'])
    if 'column' in kwargs:
        NL = kwargs['column']
    if 'systems' in kwargs:
        systems = kwargs['systems']
        sys = True
        last_sys = int(len(systems) - 1)
        show_systems = []
        
        for s_chk in np.arange(len(systems)):
            if type(systems[s_chk]) != int:
                raise Exception('systems must be an array of all integers')
        
        if 'show_systems' in kwargs:
            show_systems = kwargs['show_systems']
            if len(systems) != len(show_systems):
                raise Exception('systems and show_systems need to be arrays of equal length')
            
            for ls in np.arange(len(show_systems)):
                if show_systems[ls] not in [True, False]:
                    raise Exception('show_systems must be an array of Truth Values')
                if show_systems[ls] == True:
                    last_sys = int(ls)
        else:
            for ss in np.arange(len(systems)):
                show_systems.append(True)
    
    wavefunction = ''
    qubits = int(m.log(len(statevec), 2))
    
    for i in range(int(len(statevec))):
        value = round(statevec[i].real, dec) + round(statevec[i].imag, dec) * 1j
        if (value.real != 0) or (value.imag != 0):
            state = list(Binary(int(i), int(2**qubits), 'L'))
            state_str = ''
            
            if sys == True:
                k = 0
                for s in np.arange(len(systems)):
                    if show_systems[s] == True:
                        if int(s) != last_sys:
                            state.insert(int(k + systems[s]), '>|')
                            k = int(k + systems[s] + 1)
                        else:
                            k = int(k + systems[s])
                    else:
                        for s2 in np.arange(systems[s]):
                            del state[int(k)]
            
            for j in np.arange(len(state)):
                if type(state[j]) != str:
                    state_str =  str(int(state[j]))  + state_str 
                else:
                    state_str =   state[j] +state_str
            
            if (value.real != 0) and (value.imag != 0):
                if value.imag > 0:
                    wavefunction = wavefunction + str(value.real) + '+' + str(value.imag) + 'j |' + state_str + '> '
                else:
                    wavefunction = wavefunction + str(value.real) + '' + str(value.imag) + 'j |' + state_str + '> '
            
            if (value.real != 0) and (value.imag == 0):
                wavefunction = wavefunction + str(value.real) + ' |' + state_str + '> '
            
            if (value.real == 0) and (value.imag != 0):
                wavefunction = wavefunction + str(value.imag) + 'j |' + state_str + '> '
            
            if NL:
                wavefunction = wavefunction + '\n'
    
    print(wavefunction)

def Measurement(quantumcircuit, **kwargs):
    '''
    Executes a measurement(s) of a QuantumCircuit object for tidier printing
    Keyword Arguments: shots (integer) - number of trials to execute for the measurement(s)
                       return_M (Bool) - indicates whether to return the Dictionary object containing measurement results
                       print_M (Bool) - indicates whether to print the measurement results
                       column (Bool) - prints each state in a vertical column
    '''
    p_M = True
    S = 1
    ret = False
    NL = False
    
    if 'shots' in kwargs:
        S = int(kwargs['shots'])
    if 'return_M' in kwargs:
        ret = kwargs['return_M']
    if 'print_M' in kwargs:
        p_M = kwargs['print_M']
    if 'column' in kwargs:
        NL = kwargs['column']
    
    M1 = execute(quantumcircuit, M_simulator, shots=S).result().get_counts(quantumcircuit)
    M2 = {}
    k1 = list(M1.keys())
    v1 = list(M1.values())
    
    for k in np.arange(len(k1)):
        key_list = list(k1[k])
        new_key = ''
        
        for j in np.arange(len(key_list)):
            new_key = new_key + key_list[len(key_list) - (j + 1)]
        
        M2[new_key] = v1[k]
    
    if p_M:
        k2 = list(M2.keys())
        v2 = list(M2.values())
        measurements = ''
        
        for i in np.arange(len(k2)):
            m_str = ''
            
            for j in np.arange(len(k2[i])):
                if k2[i][j] == '0':
                    m_str = '0' + m_str
                if k2[i][j] == '1':
                    m_str = '1'+ m_str
                if k2[i][j] == ' ':
                    m_str = m_str + '>|'
            
            m_str = str(v2[i])  + '|' + m_str + '> '
            
            if NL:
                m_str = m_str + '\n'
            
            measurements = measurements + m_str
        
        print(measurements)
    
    if ret:
        return M2


def Most_Probable(M, N):
    '''
    Input: M (Dictionary), N (integer)
    Returns the N most probable states according to the measurement counts stored in M
    '''
    count = []
    state = []
    
    if len(M) < N:
        N = len(M)
    
    for k in np.arange(N):
        count.append(0)
        state.append(0)
    
    for m in np.arange(len(M)):
        new = True
        
        for n in np.arange(N):
            if (list(M.values())[int(m)] > count[int(n)]) and new:
                for i in np.arange(int(N - (n + 1))):
                    count[-int(1 + i)] = count[-int(1 + i + 1)]
                    state[-int(1 + i)] = state[-int(1 + i + 1)]
                
                count[int(n)] = list(M.values())[m]
                state[int(n)] = list(M.keys())[m]
                new = False
    
    return count, state


def Binary(N, total, LSB):
    '''
    Input: N (integer), total (integer), LSB (string)
    Returns the base-2 binary equivalent of N according to left or right least significant bit notation
    '''
    qubits = int(m.log(total, 2))
    b_num = np.zeros(qubits)
    
    for i in np.arange(qubits):
        if N / (2 ** (qubits - i - 1)) >= 1:
            if LSB == 'R':
                b_num[i] = 1
            if LSB == 'L':
                b_num[int(qubits - (i + 1))] = 1
            
            N = N - 2 ** (qubits - i - 1)
    
    B = []
    
    for j in np.arange(len(b_num)):
        B.append(int(b_num[j]))
    
    return B


def From_Binary(S, LSB):
    '''
    Input: S (string or array), LSB (string)
    Converts a base-2 binary number to base-10 according to left or right least significant bit notation
    '''
    num = 0
    
    for i in np.arange(len(S)):
        if LSB == 'R':
            num = num + int(S[int(0 - (i + 1))]) * 2 ** (i)
        if LSB == 'L':
            num = num + int(S[int(i)]) * 2 ** (i)
    
    return num

def X_Transformation(qc, qreg, state):
    '''
    Input: qc (QuantumCircuit) qreg (QuantumRegister) state (array)
    Applies the necessary X gates to transform 'state' to the state of all 1's
    '''
    for j in np.arange(len(state)):
        if( int(state[j])==0 ):
            qc.x( qreg[int(j)] )

def n_NOT(qc, control, target, anc):
    '''
    Input: qc (QuantumCircuit) control (QuantumRegister) target (QuantumRegister[integer]) anc (QuantumRegister)
    Applies the necessary CCX gates to perform a higher order control-X operation on the target qubit
    '''
    n = len(control)
    instructions = []
    active_ancilla = []
    q_unused = []
    q = 0
    a = 0
    while( (n > 0) or (len(q_unused)!=0) or (len(active_ancilla)!=0) ):
        if( n > 0 ):
            if( (n-2) >= 0 ):
                instructions.append( [control[q], control[q+1], anc[a]] )
                active_ancilla.append(a)
                a = a + 1
                q = q + 2
                n = n - 2
            if( (n-2) == -1 ):
                q_unused.append( q )
                n = n - 1
        elif( len(q_unused) != 0 ):
            if(len(active_ancilla)!=1):
                instructions.append( [control[q], anc[active_ancilla[0]], anc[a]] )
                del active_ancilla[0]
                del q_unused[0]
                active_ancilla.append(a)
                a = a + 1
            else:
                instructions.append( [control[q], anc[active_ancilla[0]], target] )
                del active_ancilla[0]
                del q_unused[0]
        elif( len(active_ancilla)!=0 ):
            if( len(active_ancilla) > 2 ):
                instructions.append( [anc[active_ancilla[0]], anc[active_ancilla[1]], anc[a]] )
                active_ancilla.append(a)
                del active_ancilla[0]
                del active_ancilla[0]
                a = a + 1
            elif( len(active_ancilla)==2):
                instructions.append([anc[active_ancilla[0]], anc[active_ancilla[1]], target])
                del active_ancilla[0]
                del active_ancilla[0]
        
    for i in np.arange( len(instructions) ):
        qc.ccx( instructions[i][0], instructions[i][1], instructions[i][2] )
    del instructions[-1]
    
    for i in np.arange( len(instructions) ):
        qc.ccx( instructions[0-(i+1)][0], instructions[0-(i+1)][1], instructions[0-(i+1)][2] )

def n_Control_U(qc, control, anc, gates):
    '''
    Input: qc (QuantumCircuit) control (QuantumRegister) anc (QuantumRegister)
    gates (array of the form [[string,QuantumRegister[i]],[],...])
    Performs the list of control gates on the respective target qubits as a higher order N-control operation
    '''
    if len(gates) != 0:
        instructions = []
        active_ancilla = []
        q_unused = []
        n = len(control)
        q = 0
        a = 0
        while n > 0 or len(q_unused) != 0 or len(active_ancilla) != 0:
            if n > 0:
                if (n - 2) >= 0:
                    instructions.append([control[q], control[q + 1], anc[a]])
                    active_ancilla.append(a)
                    a = a + 1
                    q = q + 2
                    n = n - 2
                if (n - 2) == -1:
                    q_unused.append(q)
                    n = n - 1
            elif len(q_unused) != 0:
                if len(active_ancilla) > 1:
                    instructions.append([control[q], anc[active_ancilla[0]], anc[a]])
                    del active_ancilla[0]
                    del q_unused[0]
                    active_ancilla.append(a)
                    a = a + 1
                else:
                    instructions.append([control[q], anc[active_ancilla[0]], anc[a]])
                    del active_ancilla[0]
                    del q_unused[0]
                    c_a = anc[a]
            elif len(active_ancilla) != 0:
                if len(active_ancilla) > 2:
                    instructions.append([anc[active_ancilla[0]], anc[active_ancilla[1]], anc[a]])
                    active_ancilla.append(a)
                    del active_ancilla[0]
                    del active_ancilla[0]
                    a = a + 1
                elif len(active_ancilla) == 2:
                    instructions.append([anc[active_ancilla[0]], anc[active_ancilla[1]], anc[a]])
                    del active_ancilla[0]
                    del active_ancilla[0]
                    c_a = anc[a]
                elif len(active_ancilla) == 1:
                    c_a = anc[active_ancilla[0]]
                    del active_ancilla[0]
        
        for i in np.arange(len(instructions)):
            qc.ccx(instructions[i][0], instructions[i][1], instructions[i][2])
        
        for j in np.arange(len(gates)):
            control_vec = [gates[j][0], c_a]
            for k in np.arange(1, len(gates[j])):
                control_vec.append(gates[j][k])
            if control_vec[0] == 'X':
                qc.cx(control_vec[1], control_vec[2])
            if control_vec[0] == 'Z':
                qc.cz(control_vec[1], control_vec[2])
            if control_vec[0] == 'PHASE':
                qc.cp(control_vec[3], control_vec[1], control_vec[2])
            if control_vec[0] == 'SWAP':
                qc.cswap(control_vec[1], control_vec[2], control_vec[3])
        
        for i in np.arange(len(instructions)):
            qc.ccx(instructions[0 - (i + 1)][0], instructions[0 - (i + 1)][1], instructions[0 - (i + 1)][2])
def Blackbox_D(qc,qcreg):
    type = random.choice(['balanced','constant'])
    if type == 'constant':
        if random.randrange(2)==0:
            qc.x(qcreg[1])
    elif type == 'balanced':
        qc.cx(qcreg[0],qcreg[1])
        if random.randrange(2)==0:
            qc.x(qcreg[1])
    return (qc,type)
def Blackbox_DJ(qc,qcreg):
    type = random.choice(['other','constant'])
    n = len(qcreg)
    if type == 'constant':
        if random.randrange(2)==0:
            qc.x(qcreg[n-1])
    elif type == 'other':
        x = random.randrange(n)
        for i in range(x):
            c = random.randrange(n-1)
            qc.cx(qcreg[c],qcreg[n-1])
        if random.randrange(2)==0:
            qc.x(qcreg[1])
    return (qc,type)
def Blackbox_g_DJ(Q,qc,qreg,an1):
    f =[]
    type = random.choice(['balanced','constant'])
    if type == 'constant':
        if random.randrange(2)==0:
            qc.x(qreg)
        f.append(type)
    else:
        an2 = QuantumRegister(int(Q-2))
        Qc = QuantumCircuit(an2)
        qc += Qc
        
        S = []
        for s in range(2**Q):
            S.append(s)
        

def QFT(qc, q, qubits, **kwargs):
    '''
    Input: qc (QuantumCircuit) q (QuantumRegister) qubits (integer)
    Keyword Arguments: swap (Bool) - Adds SWAP gates after all of the phase gates have been applied
    Assigns all the gate operations for a Quantum Fourier Transformation
    '''
    R_phis = [0]
    for i in np.arange(2, int(qubits + 1)):
        R_phis.append(2 / (2**(i)) * m.pi)
    
    for j in np.arange(int(qubits)):
        qc.h(q[int(j)])
        
        for k in np.arange(int(qubits - (j + 1))):
            qc.cu1(R_phis[k + 1], q[int(j + k + 1)], q[int(j)])
    
    if 'swap' in kwargs:
        if kwargs['swap'] == True:
            for s in np.arange(m.floor(qubits / 2.0)):
                qc.swap(q[int(s)], q[int(qubits - 1 - s)])

def QFT_dgr(qc, q, qubits, **kwargs):
    '''
    Input: qc (QuantumCircuit) q (QuantumRegister) qubits (integer)
    Keyword Arguments: swap (Bool) - Adds SWAP gates after all of the phase gates have been applied
    Assigns all the gate operations for a Quantum Fourier Transformation
    '''
    if 'swap' in kwargs:
        if kwargs['swap'] == True:
            for s in np.arange(m.floor(qubits / 2.0)):
                qc.swap(q[int(s)], q[int(qubits - 1 - s)])
    
    R_phis = [0]
    for i in np.arange(2, int(qubits + 1)):
        R_phis.append(-2 / (2**(i)) * m.pi)
    
    for j in np.arange(int(qubits)):
        for k in np.arange(int(j)):
            qc.cu1(R_phis[int(j - k)], q[int(qubits - (k + 1))], q[int(qubits - (j + 1))])
        qc.h(q[int(qubits - (j + 1))])

def DFT(x, **kwargs):
    '''
    Input: x (array)
    Keyword Arguments: inverse (Bool) - if True, performs an Inverse Discrete Fourier Transformation instead
    Computes a classical Discrete Fourier Transformation on the array of values x, returning a new array of transformed values
    '''
    p = -1.0
    if 'inverse' in kwargs:
        P = kwargs['inverse']
        if P == True:
            p = 1.0
    
    L = len(x)
    X = []
    
    for i in np.arange(L):
        value = 0
        for j in np.arange(L):
            value = value + x[j] * np.exp(p * 2 * m.pi * 1.0j * (int(i * j) / (L * 1.0)))
        X.append(value)
    
    for k in np.arange(len(X)):
        re = round(X[k].real, 5)
        im = round(X[k].imag, 5)
        
        if (abs(im) == 0) and (abs(re) != 0):
            X[k] = re
        elif (abs(re) == 0) and (abs(im) != 0):
            X[k] = im * 1.0j
        elif (abs(re) == 0) and (abs(im) == 0):
            X[k] = 0
        else:
            X[k] = re + im * 1.0j
    
    return X

#==========================================================
#----------- Lesson 6.1 - Quantum Adder ------------
#==========================================================

def Quantum_Adder(qc, Qa, Qb, A, B):
    '''
    Input: qc (QuantumCircuit) Qa (QuantumRegister) Qb (QuantumRegister) A (array) B (array)
    Appends all of the gate operations for a QFT based addition of two states A and B
    '''
    Q = len(B)
    
    for n in np.arange(Q):
        if A[n] == 1:
            qc.x(Qa[int(n + 1)])
        if B[n] == 1:
            qc.x(Qb[int(n)])
    
    QFT(qc, Qa, Q + 1)
    p = 1
    
    for j in np.arange(Q):
        qc.cu1(m.pi / (2**p), Qb[int(j)], Qa[0])
        p = p + 1
    
    for i in np.arange(1, Q + 1):
        p = 0
        for jj in np.arange(i - 1, Q):
            qc.cu1(m.pi / (2**p), Qb[int(jj)], Qa[int(i)])
            p = p + 1
    
    QFT_dgr(qc, Qa, Q + 1)
def QPE_phi(MP):
    '''
    Input: array( [[float,float],[string,string]] )
    Takes in the two most probable states and their probabilities, returns phi and the approximate theta for QPE
    '''
    ms = [[], []]
    
    for i in np.arange(2):
        for j in np.arange(len(MP[1][i])):
            ms[i].append(int(MP[1][i][j]))
    
    n = int(len(ms[0]))
    MS1 = From_Binary(ms[0], 'R')
    MS2 = From_Binary(ms[1], 'R')
    PHI = [99, 0]
    
    for k in np.arange(1, 5000):
        phi = k / 5000
        prob = 1 / (2**(2*n)) * abs((-1 + np.exp(2.0j * m.pi * phi)) / (-1 + np.exp(2.0j * m.pi * phi / (2**n))))**2
        
        if abs(prob - MP[0][0]) < abs(PHI[0] - MP[0][0]):
            PHI[0] = prob
            PHI[1] = phi
    
    if (MS1 < MS2) and ((MS1 != 0) and (MS2 != (2**n - 1))):
        theta = (MS1 + PHI[1]) / (2**n)
    elif (MS1 > MS2) and (MS1 != 0):
        theta = (MS1 - PHI[1]) / (2**n)
    else:
        theta = 1 + (MS1 - PHI[1]) / (2**n)
    
    return PHI[1], theta

#=============================================================
#----------- Lesson 7.1 - Quantum Counting ------------
#=============================================================

def Grover_Oracle(mark, qc, q, an1, an2):
    '''
    Input: mark (array) qc (QuantumCircuit) q (QuantumRegister) an1 (QuantumRegister) an2 (QuantumRegister)
    Appends the necessary gates for a phase flip on the marked state
    '''
    qc.h(an1[0])
    X_Transformation(qc, q, mark)
    
    if len(mark) > 2:
        n_NOT(qc, q, an1[0], an2)
    
    if len(mark) == 2:
        qc.ccx(q[0], q[1], an1[0])
    
    X_Transformation(qc, q, mark)
    qc.h(an1[0])

def Grover_Diffusion(mark, qc, q, an1, an2):
    '''
    Input: mark (array) qc (QuantumCircuit) q (QuantumRegister) an1 (QuantumRegister) an2 (QuantumRegister)
    Appends the necessary gates for a Grover Diffusion operation
    '''
    zeros_state = []
    
    for i in np.arange(len(mark)):
        zeros_state.append(0)
    
    for i in np.arange(len(mark)):
        qc.h(q[int(i)])
    
    Grover_Oracle(zeros_state, qc, q, an1, an2)
    
    for j in np.arange(len(mark)):
        qc.h(q[int(j)])

def Multi_Grover(q, a1, a2, qc, marked, iters):
    '''
    Input: q (QuantumRegister) a1 (QuantumRegister) a2 (QuantumRegister) qc (QuantumCircuit)
    marked (array) iters (integer)
    Appends all of the gate operations for a multi-marked state Grover Search
    '''
    Q = int(len(marked))
    
    for i in np.arange(iters):
        for j in np.arange(len(marked)):
            M = list(marked[j])
            
            for k in np.arange(len(M)):
                if M[k] == '1':
                    M[k] = 1
                else:
                    M[k] = 0
            
            Grover_Oracle(M, qc, q, a1, a2)
            Grover_Diffusion(M, qc, q, a1, a2)
    
    return qc, q, a1, a2
def C_Oracle(qc, c, q, a1, a2, state):
    '''
    Input: qc (QuantumCircuit) c (QuantumRegister[i]) q (QuantumRegister)
    a1 (QuantumRegister) a2 (QuantumRegister) state (array)
    Appends all of the gate operations for a control-Grover Oracle operation
    '''
    N = len(state)
    
    for i in np.arange(N):
        if state[i] == 0:
            qc.cx(c, q[int(i)])
    
    #---------------------------------
    qc.ccx(q[0], q[1], a1[0])
    
    for j1 in np.arange(N-2):
        qc.ccx(q[int(2+j1)], a1[int(j1)], a1[int(1+j1)])
    
    qc.ccx(c, a1[N-2], a2[0])
    
    for j2 in np.arange(N-2):
        qc.ccx(q[int(N-1-j2)], a1[int(N-3-j2)], a1[int(N-2-j2)])
    
    qc.ccx(q[0], q[1], a1[0])
    
    #---------------------------------
    for i2 in np.arange(N):
        if state[i2] == 0:
            qc.cx(c, q[int(i2)])

def C_Diffusion(qc, c, q, a1, a2, Q, ref):
    '''
    Input: qc (QuantumCircuit) c (QuantumRegister[i]) q (QuantumRegister) a1 (QuantumRegister)
    a2 (QuantumRegister) Q (integer) ref (Bool)
    Appends all of the gate operations for a control-Grover Diffusion operation
    '''
    N = 2**(Q)
    
    for j in np.arange(Q):
        qc.ch(c, q[int(j)])
    
    if ref:
        for k in np.arange(1, N):
            C_Oracle(qc, c, q, a1, a2, Binary(int(k), N, 'R'))
    else:
        C_Oracle(qc, c, q, a1, a2, Binary(0, N, 'R'))
    
    for j2 in np.arange(Q):
        qc.ch(c, q[int(j2)])

def C_Grover(qc, c, q, a1, a2, marked, **kwargs):
    '''
    Input: qc (QuantumCircuit) c (QuantumRegister[i]) q (QuantumRegister)
    a1 (QuantumRegister) a2 (QuantumRegister) marked (array)
    Keyword Arguments: proper (Bool) - Dictates how to perform the reflection about the average within the Diffusion Op
    Appends all of the gate operations for a control-Grover
    '''
    Reflection = False
    
    if 'proper' in kwargs:
        Reflection = kwargs['proper']
    
    M = []
    
    for m1 in np.arange(len(marked)):
        M.append(list(marked[m1]))
    
    for m2 in np.arange(len(M[m1])):
        M[m1][m2] = int(M[m1][m2])
    
    L = len(M[0])
    N = 2**(L)
    
    for i in np.arange(len(M)):
        C_Oracle(qc, c, q, a1, a2, M[i])
    
    C_Diffusion(qc, c, q, a1, a2, L, Reflection)

#================================================
#----------- Lesson 8 - Shor's -----------
#================================================

def GCD(a, b):
    '''
    Input: a (integer) b (integer)
    Computes the greatest common denominator between a and b using an inefficient exhasutive search
    '''
    gcd = 0
    
    if a > b:
        num1 = a
        num2 = b
    elif b > a:
        num1 = b
        num2 = a
    elif a == b:
        gcd = a
    
    while gcd == 0:
        i = 1
        
        while num1 >= num2*i:
            i = i + 1
            
        if num1 == num2*(i-1):
            gcd = num2
        else:
            r = num1 - num2*(i-1)
            num1 = num2
            num2 = r
    
    return gcd

def Euclids_Alg(a, b):
    '''
    Input: a (integer) b (integer)
    Computes the greatest common denominator between a and b using Euclid's Algorithm
    '''
    if a > b:
        num1 = a
        num2 = b
    if b > a:
        num1 = b
        num2 = a
    if b == a:
        gcd = a
        
    r_old = 0
    r_new = int(num1 % num2)
    r_old = int(num2)
    
    while r_new != 0:
        r_old = r_new
        r_new = int(num1 % num2)
        num1 = num2
        num2 = r_new
        
    gcd = r_old
    return gcd

def Modulo_f(Q, a, N):
    '''
    Input: Q (integer) a (integer) N (integer)
    Produces an array of all the final modulo N results for the power function a^x (mod N)
    '''
    mods = [1]
    
    for i in np.arange(1, 2**Q):
        if i == 1:
            mods.append(a**i % N)
            num = a**i % N
        if i > 1:
            mods.append((num * a) % N)
            num = (num * a) % N
    
    return mods

def Mod_Op(Q, qc, q1, q2, anc, a, N):
    '''
    Input: Q (integer) qc (QuantumCircuit) q1 (QuantumRegister) q2 (QuantumRegister)
    anc (QuantumRegister) a (integer) N (integer)
    Applies the Modulo Multiplication operator for Shor's algorithm
    '''
    mods = Modulo_f(Q, a, N)
    
    for j in np.arange(2**Q):
        q1_state = Binary(j, 2**Q, 'L')
        q2_state = Binary(mods[j], 2**Q ,'L')
        gates = []
        
        for k in np.arange(Q):
            if q2_state[k] == 1:
                gates.append(['X', q2[int(k)]])
        
        X_Transformation(qc, q1, q1_state)
        n_Control_U(qc, q1, anc, gates)
        X_Transformation(qc, q1, q1_state)

def ConFrac(N, **kwargs):
    '''
    Input: N (float)
    Keyword Arguments: a_max (integer) - the maximum number of iterations to continue approximating
                       return_a (Bool) - if True, returns the array a containing the continued fraction information
    Evaluates the non-integer number N as the quantity p/q, where p and q are integers
    '''
    imax = 20
    r_a = False
    
    if 'a_max' in kwargs:
        imax = kwargs['a_max']
    if 'return_a' in kwargs:
        r_a = kwargs['return_a']
    
    a = []
    a.append(m.floor(N))
    b = N - a[0]
    i = 1
    
    while (round(b, 10) != 0) and (i < imax):
        n = 1.0 / b
        a.append(m.floor(n))
        b = n - a[-1]
        i = i + 1
    
    #------------------------------
    a_copy = []
    
    for ia in np.arange(len(a)):
        a_copy.append(a[ia])
    
    for j in np.arange(len(a) - 1):
        if j == 0:
            p = a[-1] * a[-2] + 1
            q = a[-1]
            del a[-1]
            del a[-1]
        else:
            p_new = a[-1] * p + q
            q_new = p
            p = p_new
            q = q_new
            del a[-1]
    
    if r_a == True:
        return q, p, a_copy
    else:
        return q, p
def r_Finder(a, N):
    '''
    Input: a (integer) N (integer)
    Exhaustively computes the period r to the modulo power function a^x (mod N)
    '''
    value1 = a**1 % N
    r = 1
    value2 = 0
    
    while (value1 != value2) or (r > 1000):
        value2 = a**(int(1 + r)) % N
        
        if value1 != value2:
            r = r + 1
            
    return r

def Primality(N):
    '''
    Input: N (integer)
    Returns True if N is a prime number, otherwise False
    '''
    is_prime = True
    
    if (N == 1) or (N == 2) or (N == 3):
        is_prime = True
    elif (N % 2 == 0) or (N % 3 == 0):
        is_prime = False
    elif is_prime == True:
        p = 5
        
        while (p**2 <= N) and (is_prime == True):
            if (N % p == 0) or (N % (p + 2) == 0):
                is_prime = False
            p = p + 6
            
    return is_prime

def Mod_r_Check(a, N, r):
    '''
    Input: a (integer) N (integer) r (integer)
    Checks a value of r, returning True or False based on whether it correctly leads to a factor of N
    '''
    v1 = a**(int(2)) % N
    v2 = a**(int(2 + r)) % N
    
    if (v1 == v2) and (r < N) and (r != 0):
        return True
    else:
        return False

def Evaluate_S(S, L, a, N):
    '''
    Input: S (integer) L (integer) a (integer) N (integer)
    Attempts to use the measured state |S> to find the period r
    '''
    Pairs = [[S, L]]
    
    for s in np.arange(3):
        S_new = int(S - 1 + s)
        
        for l in np.arange(3):
            L_new = int(L - 1 + l)
            
            if (S_new != S) or (L_new != L) and (S_new != L_new):
                Pairs.append([S_new, L_new])
    
    #--------------------------- Try 9 combinations of S and L, plus or minus 1 from S & L
    period = 0
    r_attempts = []
    found_r = False
    
    while (found_r == False) and (len(Pairs) != 0):
        order = 1
        S_o = Pairs[0][0]
        160
        L_o = Pairs[0][1]
        q_old = -1
        q = 999
        
        while q_old != q:
            q_old = int(q)
            q, p = ConFrac(S_o / L_o, a_max=(order + 1))
            new_r = True
            
            for i in np.arange(len(r_attempts)):
                if q == r_attempts[i]:
                    new_r = False
                    
            if new_r:
                r_attempts.append(int(q))
                r_bool = Mod_r_Check(a, N, q)
                
                if r_bool:
                    found_r = True
                    q_old = q
                    period = int(q)
                    
            order = order + 1
            del Pairs[0]
    
    #--------------------------- Try higher multiples of already attempted r values
    r_o = 0
    
    while (found_r == False) and (r_o < len(r_attempts)):
        k = 2
        r2 = r_attempts[r_o]
        
        while k * r2 < N:
            r_try = int(k * r2)
            new_r = True
            
            for i2 in np.arange(len(r_attempts)):
                if r_try == r_attempts[i2]:
                    new_r = False
                    
            if new_r:
                r_attempts.append(int(r_try))
                r_bool = Mod_r_Check(a, N, r_try)
                
                if r_bool:
                    found_r = True
                    k = N
                    period = int(r_try)
                    
            k = k + 1
            r_o = r_o + 1
    
    #--------------------------- If a period is found, try factors of r for smaller periods
    if found_r == True:
        Primes = []
        
        for i in np.arange(2, period):
            if Primality(int(i)):
                Primes.append(int(i))
                
        if len(Primes) > 0:
            try_smaller = True
            
            while try_smaller == True:
                found_smaller = False
                p2 = 0
                
                while (found_smaller == False) and (p2 < len(Primes)):
                    try_smaller = False
                    
                    if period / Primes[p2] == m.floor(period / Primes[p2]):
                        r_bool_2 = Mod_r_Check(a, N, int(period / Primes[p2]))
                        
                        if r_bool_2:
                            period = int(period / Primes[p2])
                            found_smaller = True
                            try_smaller = True
                            
                    p2 = p2 + 1
    
    return period
def k_Data(k, n):
    '''
    Input: k (integer) n (integer)
    Creates a random set of data loosely centered around k locations
    '''
    Centers = []
    
    for i in np.arange(k):
        Centers.append([1.5 + np.random.rand() * 5, 1.5 * random.random() * 5])
        
    count = round((0.7 * n) / k)
    Data = []
    
    for j in np.arange(len(Centers)):
        for j2 in np.arange(count):
            r = random.random() * 1.5
            Data.append([Centers[j][0] + r * np.cos(random.random() * 2 * m.pi),
                         Centers[j][1] + r * np.sin(random.random() * 2 * m.pi)])
    
    diff = int(n - k * count)
    
    for j2 in np.arange(diff):
        Data.append([random.random() * 8, random.random() * 8])
    
    return Data


def Initial_Centroids(k, D):
    '''
    Input: k (integer) D (array)
    Picks k data points at random from the list D
    '''
    D_copy = []
    
    for i in np.arange(len(D)):
        D_copy.append(D[i])
        
    Centroids = []
    
    for j in np.arange(k):
        p = random.randint(0, int(len(D_copy) - 1))
        Centroids.append([D_copy[p][0], D_copy[p][1]])
        D_copy.remove(D_copy[p])
        
    return Centroids


def Update_Centroids(CT, CL):
    '''
    Input: CT (array) CL (array)
    Based on the data within each cluster, computes and returns new Centroids using mean coordinate values
    '''
    old_Centroids = []
    
    for c0 in np.arange(len(CT)):
        old_Centroids.append(CT[c0])
        
    Centroids = []
    
    for c1 in np.arange(len(CL)):
        mean_x = 0
        mean_y = 0
        
        for c2 in np.arange(len(CL[c1])):
            mean_x = mean_x + CL[c1][c2][0] / len(CL[c1])
            mean_y = mean_y + CL[c1][c2][1] / len(CL[c1])
            
        Centroids.append([mean_x, mean_y])
        
    return Centroids, old_Centroids


def Update_Clusters(D, CT, CL):
    '''
    Input: D (array) CT (array) CL (array)
    Using all data points and Centroids, computes and returns the new array of Clusters
    '''
    old_Clusters = []
    
    for c0 in np.arange(len(CL)):
        old_Clusters.append(CL[c0])
        
    Clusters = []
    
    for c1 in np.arange(len(CT)):
        Clusters.append([])
        
    for d in np.arange(len(D)):
        closest = 'c'
        distance = 100000
        
        for c2 in np.arange(len(Clusters)):
            Dist = m.sqrt((CT[c2][0] - D[d][0])**2 + (CT[c2][1] - D[d][1])**2)
            
            if Dist < distance:
                distance = Dist
                closest = int(c2)
                
        Clusters[closest].append(D[d])
        
    return Clusters, old_Clusters


def Check_Termination(CL, oCL):
    '''
    Input: CL (array) oCL (array)
    Returns True or False based on whether the Update_Clusters function has caused any data points to change clusters
    '''
    terminate = True
    
    for c1 in np.arange(len(oCL)):
        for c2 in np.arange(len(oCL[c1])):
            P_found = False
            
            for c3 in np.arange(len(CL[c1])):
                if CL[c1][c3] == oCL[c1][c2]:
                    P_found = True
                    
            if P_found == False:
                terminate = False
                
    return terminate

def Draw_Data(CL, CT, oCT, fig, ax, colors, colors2):
    '''
    Input: CL (array) CT (array) oCT (array) fig (matplotlib figure) ax (figure subplot)
    colors (array of color strings) colors2 (array of color strings)
    Using the arrays Clusters, Centroids, and old Centroids, draws and colors each data point according to its cluster
    '''
    for j1 in np.arange(len(CL)):
        ax.scatter(oCT[j1][0], oCT[j1][1], color='white', marker='s', s=80)
        
    for cc in np.arange(len(CL)):
        for ccc in np.arange(len(CL[cc])):
            ax.scatter(CL[cc][ccc][0], CL[cc][ccc][1], color=colors[cc], s=10)
            
    for j2 in np.arange(len(CL)):
        ax.scatter(CT[j2][0], CT[j2][1], color=colors2[j2], marker='x', s=50)
        
    fig.canvas.draw()
    time.sleep(1)


def SWAP_Test(qc, control, q1, q2, classical, S):
    '''
    Input: qc (QuantumCircuit) control (QuantumRegister[i]) q1 (QuantumRegister[i]) q2 (QuantumRegister[i])
    classical (ClassicalRegister[i]) S (integer)
    Appends the necessary gates for 2-Qubit SWAP Test and returns the number of |0> state counts
    '''
    qc.h(control)
    qc.cswap(control, q1, q2)
    qc.h(control)
    qc.measure(control, classical)
    D = {'0': 0}
    D.update(Measurement(qc, shots=S, return_M=True, print_M=False))
    return D['0']


def Bloch_State(p, P):
    '''
    Input: p (array) P(array)
    Returns the corresponding theta and phi values of the data point p, according to min / max paramters of P
    '''
    x_min = P[0]
    x_max = P[1]
    y_min = P[2]
    y_max = P[3]
    theta = np.pi / 2 * ((p[0] - x_min) / (1.0 * x_max - x_min) + (p[1] - y_min) / (1.0 * y_max - y_min))
    phi = np.pi / 2 * ((p[0] - x_min) / (1.0 * x_max - x_min) - (p[1] - y_min) / (1.0 * y_max - y_min) + 1)
    return theta, phi


def Q_Update_Clusters(D, CT, CL, DS, shots):
    '''
    Input: D (array) CT (array) CL (array) DS (array) shots (integer)
    Using all data points, Centroids, uses the SWAP Test to compute and return the new array of Clusters
    '''
    old_Clusters = []
    
    for c0 in np.arange(len(CL)):
        old_Clusters.append(CL[c0])
        
    Clusters = []
    
    for c1 in np.arange(len(CT)):
        Clusters.append([])
        
    #------------------------------------------------
    for d in np.arange(len(D)):
        closest = 'c'
        distance = 0
        t, p = Bloch_State(D[d], DS)
        
        for c2 in np.arange(len(Clusters)):
            t2, p2 = Bloch_State(CT[c2], DS)
            q = QuantumRegister(3, name='q')
            c = ClassicalRegister(1, name='c')
            qc = QuantumCircuit(q, c, name='qc')
            qc.u3(t, p, 0, q[1])
            qc.u3(t2, p2, 0, q[2])
            IP = SWAP_Test(qc, q[0], q[1], q[2], c[0], shots)
            
            if IP > distance:
                distance = IP
                closest = int(c2)
                
        Clusters[closest].append(D[d])
        
    return Clusters, old_Clusters


def Heatmap(data, show_text, show_ticks, ax, cmap, cbarlabel, **kwargs):
    '''
    Input: data (array) show_text (Bool) show_ticks (Bool) ax (Matplotlib subplot) cmap (string) cbarlabel (string)
    Takes in data and creates a 2D Heatmap
    '''
    valfmt = "{x:.1f}"
    textcolors = ["black", "white"]
    threshold = None
    cbar_kw = {}
    
    #----------------------------
    if not ax:
        ax = plt.gca()
        
    im = ax.imshow(data, cmap=cmap, **kwargs)
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")
    ax.grid(which="minor", color="black", linestyle='-', linewidth=1)
    
    if show_ticks:
        ax.set_xticks(np.arange(data.shape[1]))
        ax.set_yticks(np.arange(data.shape[0]))
        ax.tick_params(which="minor", bottom=False, left=False)
        
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max()) / 2.
        
    kw = dict(horizontalalignment="center", verticalalignment="center")
    
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)
        
    if show_text:
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                kw.update(color=textcolors[int(im.norm(data[i, j]) < threshold)])
                text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)

def E_Expectation_Value(qc, Energies):
    '''
    Input: qc (QuantumCircuit) Energies (array)
    Computes and returns the energy expectation value using the quantum system's wavefunction
    '''
    SV = execute(qc, S_simulator, shots=1).result().get_statevector()
    EV = 0
    for i in np.arange(len(SV)):
        EV = EV + Energies[i] * abs(SV[i] * np.conj(SV[i]))
    EV = round(EV, 4)
    return EV


def Top_States(States, Energies, SV, top):
    '''
    Input: States (array) Energies (array) SV (Qiskit statevector) top (integer)
    Displays the top most probable states in the system, and their associated energy
    '''
    P = []
    S = []
    E = []
    for a in np.arange(top):
        P.append(-1)
        S.append('no state')
        E.append('no energy')
        
    for i in np.arange(len(States)):
        new_top = False
        probs = abs(SV[i] * np.conj(SV[i])) * 100
        state = States[i]
        energ = Energies[i]
        j = 0
        
        while ((new_top == False) and (j < top)):
            if (probs > P[j]):
                for k in np.arange(int(len(P) - (j + 1))):
                    P[int(-1 - k)] = P[int(-1 - (k + 1))]
                    S[int(-1 - k)] = S[int(-1 - (k + 1))]
                    E[int(-1 - k)] = E[int(-1 - (k + 1))]
                    
                P[j] = probs
                S[j] = state
                E[j] = energ
                new_top = True
            j = int(j + 1)
            
    for s in np.arange(top):
        print('State ', S[s], ' Probability: ', round(P[s], 2), '%', ' Energy: ', round(E[s], 2))


def Ising_Energy(V, E, **kwargs):
    '''
    Input: V (array) E (array)
    Keyword Arguments: Transverse (Bool) - Changes to the Transverse Ising Energy Model
    Calculates and returns the energy for each state according to either of the Ising Model Energy functions
    '''
    Trans = False
    
    if 'Transverse' in kwargs:
        if (kwargs['Transverse'] == True):
            Trans = True
            
    Energies = []
    States = []
    
    for s in np.arange(2 ** len(V)):
        B = Binary(int(s), 2 ** len(V), 'L')
        B2 = []
        
        for i in np.arange(len(B)):
            if (B[i] == 0):
                B2.append(1)
            else:
                B2.append(-1)
                
        state = ''
        energy = 0
        
        for s2 in np.arange(len(B)):
            state = state + str(B[s2])
            energy = energy - V[s2][1] * B2[s2]
            
        States.append(state)
        
        for j in np.arange(len(E)):
            if (Trans == False):
                energy = energy - B2[int(E[j][0])] * B2[int(E[j][1])]
            else:
                energy = energy - B2[int(E[j][0])] * B2[int(E[j][1])] * E[j][2]
                
        Energies.append(energy)
        
    return Energies, States


def Ising_Circuit(qc, q, V, E, beta, gamma, **kwargs):
    '''
    Input: qc (QuantumCircuit) q (QuantumRegister) V (array) E (array) beta (float) gamma (float)
    Keyword Arguments: Transverse (Bool) - Changes to the Transverse Ising Energy Model
    Mixing (integer) - Denotes which mixing circuit to use for U(B,beta)
    Constructs the quantum circuit for a given geometry, using the either of the Ising Model Energy functions
    '''
    Trans = False
    
    if 'Transverse' in kwargs:
        if (kwargs['Transverse'] == True):
            Trans = True
            
    Mixer = 1
    
    if 'Mixing' in kwargs:
        Mixer = int(kwargs['Mixing'])
        
    Uc_Ising(qc, q, gamma, V)
    
    if (Mixer == 2):
        Ub_Mixer2(qc, q, beta, V)
    else:
        Ub_Mixer1(qc, q, beta, V)

def Uc_Ising(qc, q, gamma, Vert, Edge, T):
    '''
    Input: qc (QuantumCircuit) q (QuantumRegister) gamma (float) Vert (array) Edge (array) T (Bool)
    Applies the necessary gates for either of the Ising Energy Model U(C,gamma) operations
    '''
    for e in np.arange(len(Edge)):  # ZZ
        if T == False:
            G = gamma
        else:
            G = gamma * Edge[e][2]
            
        qc.cx(q[int(Edge[e][0])], q[int(Edge[e][1])])
        qc.rz(2 * G, q[int(Edge[e][1])])
        qc.cx(q[int(Edge[e][0])], q[int(Edge[e][1])])
        
    for v in np.arange(len(Vert)):  # Z_gamma
        qc.rz(gamma, q[int(Vert[v][0])])


def Ub_Mixer1(qc, q, beta, Vert):
    '''
    Input: qc (QuantumCircuit) q (QuantumRegister) beta (float) Vert (array)
    Applies the necessary gates for a U(B,beta) operation using only Rx gates
    '''
    for v in np.arange(len(Vert)):
        qc.rx(beta, q[int(v)])


def Ub_Mixer2(qc, q, beta, Vert):
    '''
    Input: qc (QuantumCircuit) q (QuantumRegister) beta (float) Vert (array)
    Applies the necessary gates for a U(B,beta) operation using Rx, Ry, and CNOT gates
    '''
    for v in np.arange(len(Vert)):
        qc.rx(beta, q[int(Vert[v][0])])
        
    qc.cx(q[0], q[1])
    qc.cx(q[2], q[0])
    
    for v2 in np.arange(len(Vert)):
        qc.ry(beta, q[int(Vert[v2][0])])


def Ising_Gradient_Descent(qc, q, Circ, V, E, beta, gamma, epsilon, En, step, **kwargs):
    '''
    Input: qc (QuantumCircuit) q (QuantumRegister) Circ (Ising_Circuit function) V (array) E (array)
    beta (float) gamma (float) epsilon (float) En (array) step (float)
    Keyword Arguments: Transverse (Bool) - Changes to the Transverse Ising Energy Model
    Mixing (integer) - Denotes which mixing circuit to use for U(B,beta)
    Calculates and returns the next values for beta and gamma using gradient descent
    '''
    Trans = False
    
    if 'Transverse' in kwargs:
        if kwargs['Transverse'] == True:
            Trans = True
            
    Mixer = 1
    
    if 'Mixing' in kwargs:
        Mixer = int(kwargs['Mixing'])
        
    params = [[beta + epsilon, gamma], [beta - epsilon, gamma], [beta, gamma + epsilon], [beta, gamma - epsilon]]
    ev = []
    
    for i in np.arange(4):
        q = QuantumRegister(len(V))
        qc = QuantumCircuit(q)
        
        for hh in np.arange(len(V)):
            qc.h(q[int(hh)])
            
        Circ(qc, q, V, E, params[i][0], params[i][1], Transverse=Trans, Mixing=Mixer)
        ev.append(E_Expectation_Value(qc, En))
        
    beta_next = beta - (ev[0] - ev[1]) / (2.0 * epsilon) * step
    gamma_next = gamma - (ev[2] - ev[3]) / (2.0 * epsilon) * step
    
    return beta_next, gamma_next
def MaxCut_Energy(V, E):
    '''
    Input: V (array) E (array)
    Calculates and returns the energy for each state according to the MaxCut Energy function
    '''
    Energies = []
    States = []
    for s in np.arange(2**len(V)):
        B = Binary(int(s), 2**len(V), 'L')
        B2 = []
        for i in np.arange(len(B)):
            if B[i] == 0:
                B2.append(1)
            else:
                B2.append(-1)
        state = ''
        for s2 in np.arange(len(B)):
            state = state + str(B[s2])
        States.append(state)
        energy = 0
        for j in np.arange(len(E)):
            energy = energy + 0.5 * (1.0 - B2[int(E[j][0])] * B2[int(E[j][1])])
        Energies.append(energy)
    return Energies, States


def MaxCut_Circuit(qc, q, V, E, beta, gamma):
    '''
    Input: qc (QuantumCircuit) q (QuantumRegister) V (array) E (array) beta (float) gamma (float)
    Constructs the quantum circuit for a given geometry, using the Maxcut Energy Model
    '''
    Uc_MaxCut(qc, q, gamma, E)
    Ub_Mixer1(qc, q, beta, V)


def Uc_MaxCut(qc, q, gamma, edge):
    '''
    Input: qc (QuantumCircuit) q (QuantumRegister) gamma (float) edge (array)
    Applies the necessary gates for a U(C,gamma) operation for a MaxCut Energy Model
    '''
    for e in np.arange(len(edge)):
        qc.cx(q[int(edge[e][0])], q[int(edge[e][1])])
        qc.rz(gamma, q[int(edge[e][1])])
        qc.cx(q[int(edge[e][0])], q[int(edge[e][1])])


def p_Gradient_Ascent(qc, q, Circ, V, E, p, Beta, Gamma, epsilon, En, step):
    '''
    Input: qc (QuantumCircuit) q (QuantumRegister) Circ (MaxCut_Circuit function) V (array) E (array)
    p (integer) Beta (array) Gamma (array) epsilon (float) En (array) step (float)
    Computes the next values for beta and gamma using gradient descent, for a p-dimensional QAOA of the MaxCut Energy Model
    '''
    params = []
    for i in np.arange(2):
        for p1 in np.arange(p):
            if i == 0:
                params.append(Beta[p1])
            if i == 1:
                params.append(Gamma[p1])
    ep_params = []
    for p2 in np.arange(len(params)):
        for i2 in np.arange(2):
            ep = []
            for p3 in np.arange(len(params)):
                ep.append(params[p3])
            ep[p2] = ep[p2] + (-1.0)**(i2+1) * epsilon
            ep_params.append(ep)
    ev = []
    for p4 in np.arange(len(ep_params)):
        run_params = ep_params[p4]
        q = QuantumRegister(len(V))
        qc = QuantumCircuit(q)
        for hh in np.arange(len(V)):
            qc.h(q[int(hh)])
        for p5 in np.arange(p):
            Circ(qc, q, V, E, run_params[int(p5)], run_params[int(p5+p)])
        ev.append(E_Expectation_Value(qc, En))
    Beta_next = []
    Gamma_next = []
    for k in np.arange(len(params)):
        if k < len(params)/2:
            Beta_next.append(params[k] - (ev[int(2*k)] - ev[int(2*k+1)]) / (2.0*epsilon) * step)
        else:
            Gamma_next.append(params[k] - (ev[int(2*k)] - ev[int(2*k+1)]) / (2.0*epsilon) * step)
    return Beta_next, Gamma_next
def Single_Qubit_Ansatz(qc, qubit, params):
    '''
    Input: qc (QuantumCircuit) qubit (QuantumRegister[i]) params (array)
    Applies the necessary rotation gates for a single qubit ansatz state
    '''
    qc.ry(params[0], qubit)
    qc.rz(params[1], qubit)


def VQE_Gradient_Descent(qc, q, H, Ansatz, theta, phi, epsilon, step, **kwargs):
    '''
    Input: qc (QuantumCircuit) q (QuantumRegister) H (Dictionary) Ansatz (Single_Qubit_Ansatz function)
    theta (float) phi (float) epsilon (float) step (float)
    Keyword Arguments: measure (Bool) - Dictates whether to use measurements or the wavefunction
    shots (integer) - Dictates the number of measurements to use per computation
    Computes and returns the next values for beta and gamma using gradient descent, for a single qubit VQE
    '''
    EV_type = 'measure'
    if 'measure' in kwargs:
        M_bool = kwargs['measure']
        if M_bool == True:
            EV_type = 'measure'
        else:
            EV_type = 'wavefunction'
    Shots = 1000
    if 'shots' in kwargs:
        Shots = kwargs['shots']
    params = [theta, phi]
    ep_params = [[theta + epsilon, phi], [theta - epsilon, phi], [theta, phi + epsilon], [theta, phi - epsilon]]
    Hk = list(H.keys())
    EV = []
    for p4 in np.arange(len(ep_params)):
        H_EV = 0
        qc_params = ep_params[p4]
        for h in np.arange(len(Hk)):
            qc_params = ep_params[p4]
            q = QuantumRegister(1)
            c = ClassicalRegister(1)
            qc = QuantumCircuit(q, c)
            Ansatz(qc, q[0], [qc_params[0], qc_params[1]])
            if Hk[h] == 'X':
                qc.ry(-m.pi/2, q[0])
            if Hk[h] == 'Y':
                qc.rx(-m.pi/2, q[0])
            if EV_type == 'wavefunction':
                sv = execute(qc, S_simulator, shots=1).result().get_statevector()
                H_EV = H_EV + H[Hk[h]] * ((np.conj(sv[0]) * sv[0]).real - (np.conj(sv[1]) * sv[1]).real)
            elif EV_type == 'measure':
                qc.measure(q, c)
                M = {'0': 0, '1': 0}
                M.update(Measurement(qc, shots=Shots, print_M=False, return_M=True))
                H_EV = H_EV + H[Hk[h]] * (M['0'] - M['1']) / Shots
        EV.append(H_EV)
    theta_slope = (EV[0] - EV[1]) / (2.0 * epsilon)
    phi_slope = (EV[2] - EV[3]) / (2.0 * epsilon)
    next_theta = theta - theta_slope * step
    next_phi = phi - phi_slope * step
    return next_theta, next_phi


def Two_Qubit_Ansatz(qc, q, params):
    '''
    Input: qc (QuantumCircuit) q (QuantumRegister) params (array)
    Applies the necessary rotation and CNOT gates for a two qubit ansatz state
    '''
    Single_Qubit_Ansatz(qc, q[0], [params[0], params[1]])
    Single_Qubit_Ansatz(qc, q[1], [params[2], params[3]])
    qc.cx(q[0], q[1])
    Single_Qubit_Ansatz(qc, q[0], [params[4], params[5]])
    Single_Qubit_Ansatz(qc, q[1], [params[6], params[7]])


def Calculate_MinMax(V, C_type):
    '''
    Input: V (vert) C_type (string)
    Returns the smallest or biggest value / index for the smallest value in a list
    '''
    if C_type == 'min':
        lowest = [V[0], 0]
        for i in np.arange(1, len(V)):
            if V[i] < lowest[0]:
                lowest[0] = V[i]
                lowest[1] = int(i)
        return lowest
    if C_type == 'max':
        highest = [V[0], 0]
        for i in np.arange(1, len(V)):
            if V[i] > highest[0]:
                highest[0] = V[i]
                highest[1] = int(i)
        return highest
def Compute_Centroid(V):
    '''
    Input: V (array)
    Computes and returns the centroid from a given list of values
    '''
    points = len(V)
    dim = len(V[0])
    Cent = []
    for d in np.arange(dim):
        avg = 0
        for a in np.arange(points):
            avg = avg + V[a][d] / points
        Cent.append(avg)
    return Cent


def Reflection_Point(P1, P2, alpha):
    '''
    Input: P1 (array) P2 (array) alpha (float)
    Computes a reflection point from P1 around point P2 by an amount alpha
    '''
    P = []
    for p in np.arange(len(P1)):
        D = P2[p] - P1[p]
        P.append(P1[p] + alpha * D)
    return P


def VQE_EV(params, Ansatz, H, EV_type, **kwargs):
    '''
    Input: params (array) Ansatz( Single or Two Qubit Ansatz function) H (Dictionary) EV_type (string)
    Keyword Arguments: shots (integer) - Dictates the number of measurements to use per computation
    Computes and returns the expectation value for a given Hamiltonian and set of theta / phi values
    '''
    Shots = 10000
    if 'shots' in kwargs:
        Shots = int(kwargs['shots'])
    Hk = list(H.keys())
    H_EV = 0
    for k in np.arange(len(Hk)):
        L = list(Hk[k])
        q = QuantumRegister(len(L))
        c = ClassicalRegister(len(L))
        qc = QuantumCircuit(q, c)
        Ansatz(qc, q, params)
        sv0 = execute(qc, S_simulator, shots=1).result().get_statevector()
        if EV_type == 'wavefunction':
            for l in np.arange(len(L)):
                if L[l] == 'X':
                    qc.x(q[int(l)])
                if L[l] == 'Y':
                    qc.y(q[int(l)])
                if L[l] == 'Z':
                    qc.z(q[int(l)])
            sv = execute(qc, S_simulator, shots=1).result().get_statevector()
            H_ev = 0
            for l2 in np.arange(len(sv)):
                H_ev = H_ev + (np.conj(sv[l2]) * sv0[l2]).real
            H_EV = H_EV + H[Hk[k]] * H_ev
        elif EV_type == 'measure':
            for l in np.arange(len(L)):
                if L[l] == 'X':
                    qc.ry(-m.pi/2, q[int(l)])
                if L[l] == 'Y':
                    qc.rx(m.pi/2, q[int(l)])
            qc.measure(q, c)
            M = Measurement(qc, shots=Shots, print_M=False, return_M=True)
            Mk = list(M.keys())
            H_ev = 0
            for m1 in np.arange(len(Mk)):
                MS = list(Mk[m1])
                e = 1
                for m2 in np.arange(len(MS)):
                    if MS[m2] == '1':
                        e = e * (-1)
                H_ev = H_ev + e * M[Mk[m1]]
            H_EV = H_EV + H[Hk[k]] * H_ev / Shots
    return H_EV
def Nelder_Mead(H, Ansatz, Vert, Val, EV_type):
    '''
    Input: H (Dictionary) Ansatz( Single or Two Qubit Ansatz function) Vert (array) Val (array) EV_type (string)
    Computes and appends values for the next step in the Nelder_Mead Optimization Algorithm
    '''
    alpha = 2.0
    gamma = 2.0
    rho = 0.5
    sigma = 0.5
    add_reflect = False
    add_expand = False
    add_contract = False
    shrink = False
    add_bool = False
    # ----------------------------------------
    hi = Calculate_MinMax(Val, 'max')
    Vert2 = []
    Val2 = []
    for i in np.arange(len(Val)):
        if int(i) != hi[1]:
            Vert2.append(Vert[i])
            Val2.append(Val[i])
    Center_P = Compute_Centroid(Vert2)
    Reflect_P = Reflection_Point(Vert[hi[1]], Center_P, alpha)
    Reflect_V = VQE_EV(Reflect_P, Ansatz, H, EV_type)
    # ------------------------------------------------- # Determine if: Reflect / Expand / Contract / Shrink
    hi2 = Calculate_MinMax(Val2, 'max')
    lo2 = Calculate_MinMax(Val2, 'min')
    if hi2[0] > Reflect_V >= lo2[0]:
        add_reflect = True
    elif Reflect_V < lo2[0]:
        Expand_P = Reflection_Point(Center_P, Reflect_P, gamma)
        Expand_V = VQE_EV(Expand_P, Ansatz, H, EV_type)
        if Expand_V < Reflect_V:
            add_expand = True
        else:
            add_reflect = True
    elif Reflect_V > hi2[0]:
        if Reflect_V < hi[0]:
            Contract_P = Reflection_Point(Center_P, Reflect_P, rho)
            Contract_V = VQE_EV(Contract_P, Ansatz, H, EV_type)
            if Contract_V < Reflect_V:
                add_contract = True
            else:
                shrink = True
        else:
            Contract_P = Reflection_Point(Center_P, Vert[hi[1]], rho)
            Contract_V = VQE_EV(Contract_P, Ansatz, H, EV_type)
            if Contract_V < Val[hi[1]]:
                add_contract = True
            else:
                shrink = True
    # ------------------------------------------------- # Apply: Reflect / Expand / Contract / Shrink
    if add_reflect == True:
        new_P = Reflect_P
        new_V = Reflect_V
        add_bool = True
    elif add_expand == True:
        new_P = Expand_P
        new_V = Expand_V
        add_bool = True
    elif add_contract == True:
        new_P = Contract_P
        new_V = Contract_V
        add_bool = True
    if add_bool:
        del Vert[hi[1]]
        del Val[hi[1]]
        Vert.append(new_P)
        Val.append(new_V)
    if shrink:
        Vert3 = []
        Val3 = []
        lo = Calculate_MinMax(Val, 'min')
        Vert3.append(Vert[lo[1]])
        Val3.append(Val[lo[1]])
        for j in np.arange(len(Val)):
            if int(j) != lo[1]:
                Shrink_P = Reflection_Point(Vert[lo[1]], Vert[j], sigma)
                Vert3.append(Shrink_P)
                Val3.append(VQE_EV(Shrink_P, Ansatz, H, EV_type))
        for j2 in np.arange(len(Val)):
            del Vert[0]
            del Val[0]
            Vert.append(Vert3[j2])
            Val.append(Val3[j2])

