#!/usr/bin/env python
# coding: utf-8

# # ECE 6101 Class Project 

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import math


# # 1(a)

# In[2]:


def uniform(start,end,size):
    x = np.random.uniform(start,end,size)
    return x


# In[3]:


# uniform(0,1,100)


# # 1(b) 

# In[4]:


def cumsum(data):
    sort_data = np.array(sorted(data))
    sdata = []
    r1 = []
    problist = []
    for m in range(len(sort_data)):
        sdata.append(sort_data[m])
    temp = 0
#     plt.figure()
    for i in np.arange(0.5,1,0.01):
        count = 0
        prob = 0
        for j in range(len(sdata)):
            if sdata[j] > i:
                count +=1
        prob = count/len(sdata)
        problist.append(prob)
        r1.append(i)
    return r1,problist


# In[5]:


r1 , problist = cumsum(uniform(0,1,5000))


# In[6]:


plt.figure()
plt.plot(r1,problist)
plt.xlabel('x')
plt.ylabel('P(U>x)')
plt.grid()
plt.title('P(U>x) when x in (0.5,1)',fontsize='large',fontweight='bold')
plt.show()


# # 2(a)

# In[7]:


def ExpRand1(lambdap,data):
    U = data
    res = []
    r = []
    for i in range(len(U)):
        y = (-1/lambdap)*np.log(1-U[i]) 
        res.append(y)
        r.append(i)
    return r,res


# In[8]:


r9,res10 = ExpRand1(1/2,uniform(0,1,5000))
plt.figure()
plt.grid()
plt.plot(r9,res10)
plt.xlabel('x')
plt.ylabel('Exponential_value')
plt.title('Exponential random variable',fontsize='large',fontweight='bold')
plt.show()


# In[9]:


def ExpRand(lambdap,data):
    U = data
    res = []
    for i in range(len(U)):
        y = (-1/lambdap)*np.log(1-U[i]) 
        res.append(y)
    return res


# In[10]:


ExpRand(1/2,uniform(0,1,5000))


# In[11]:


def Poissonp(lammbda):
    L = math.exp(-lammbda)
    k = 0
    p = 1
    while p>L:
        k += 1
        p = p*np.random.uniform(0,1,1)[0]
    return k-1
    
def Poisson():
    r = []
    klist = []
    for i in range(500):
        k = Poissonp(2)
        r.append(i)
        klist.append(k)
    return r,klist


# In[12]:


r,klist = Poisson()
plt.figure()
plt.plot(r,klist)
plt.title('Poisson random variable',fontsize='large',fontweight='bold')
plt.ylabel('Poisson_value')
plt.grid()
plt.show()


# # 2(b) 

# In[13]:


def ExpPlot(lambdap,data):
    U = np.array(sorted(data))
    templist = []
    r = []
    problist = []
    for k in range(len(U)):
        y = (-lambdap)*np.log(1-U[k])
        templist.append(y)
    for i in np.arange(0,15,0.1):
        count = 0
        prob = 0
        for j in range(len(U)):
            if templist[j] > i:
                count += 1
        prob = count/len(templist)
        r.append(i)
        problist.append(np.log(prob))
    return r,problist


# In[14]:


r,klist = ExpPlot(2,uniform(0,1,10000))
plt.figure()
plt.plot(r,klist)
plt.grid()
plt.xlabel('range')
plt.ylabel('log(probablity)')
plt.title('Exponential : P(X>x)',fontsize='large',fontweight='bold')
plt.show()


# In[15]:


def Poissonp(lammbda):
    L = math.exp(-lammbda)
    k = 0
    p = 1
    while p>L:
        k += 1
        p = p*np.random.uniform(0,1,1)[0]
    return k-1
    
U = []
for k in range(500):
    k = Poissonp(2)
    U.append(k)
r = []
problist1 = []
for i in np.arange(0,15,0.1):
    count = 0 
    prob = 0
    for j in range(len(U)):
        if sorted(U)[j] > i:
            count += 1
    prob = count/len(U)
    r.append(i)
    problist1.append(np.log(prob))
plt.figure()
plt.plot(r,problist1)
plt.xlabel('range')
plt.ylabel('log(probability)')
plt.title('Poisson : P(Y>x)',fontsize='large',fontweight='bold')
plt.grid()
plt.show()


# # 3(a) 

# In[16]:


InterArrivalT = ExpRand(5,uniform(0,1,9999))


# In[17]:


ServiceT = ExpRand(6,uniform(0,1,10000))


# In[18]:


Arrival_0 = 0
Leave_0 = ServiceT[0]


# In[19]:


ArrivalQueue = [[Arrival_0,1]]


# In[20]:


temp = 0
for i in range(len(InterArrivalT)):
    temp += InterArrivalT[i]
    ArrivalQueue.append([temp,1])


# In[21]:


LeaveQueue = [[0 for col in range(2)] for row in range(10000)]
LeaveQueue[0] = [Leave_0,-1]
# LeaveQueue


# In[22]:


for i in range(1,len(ArrivalQueue)):
    if ArrivalQueue[i][0] >= LeaveQueue[i-1][0]:
        LeaveQueue[i][0] = ArrivalQueue[i][0] + ServiceT[i]
        LeaveQueue[i][1] = -1
    else:
        LeaveQueue[i][0] = LeaveQueue[i-1][0] + ServiceT[i]
        LeaveQueue[i][1] = -1


# In[23]:


TimeLine = []
templist = []
templist.extend(ArrivalQueue)
templist.extend(LeaveQueue)


# In[24]:


# templist


# In[25]:


TimeLine = sorted(templist, key=lambda state: state[0])


# In[26]:


# TimeLine


# In[27]:


state = TimeLine[0][1]
for i in range(1,len(TimeLine)):
    state += TimeLine[i][1]
    TimeLine[i][1] = state


# In[28]:


# TimeLine


# # 3(b) 

# In[29]:


maxstate = 0
for i in range(len(TimeLine)):
    if TimeLine[i][1] >= maxstate:
        maxstate = TimeLine[i][1]
maxstate


# In[30]:


def Pn(TimeLine):
    EN = 0
    templist = []
    for j in range(maxstate+1):
        timecount = 0
        for i in range(1,len(TimeLine)):
            if TimeLine[i-1][1] == j:
                timecount += (TimeLine[i][0] - TimeLine[i-1][0])
        res = timecount/TimeLine[-1][0]
        EN += j*res
        templist.append(res)
    return EN,templist


# # 3(c) 

# In[31]:


EN,rang = Pn(TimeLine)


# In[32]:


EN


# In[33]:


plt.figure()
plt.plot(range(maxstate+1),rang)
plt.xlabel('state')
plt.ylabel('P(n)')
plt.title('P(n) of M/M/1',fontsize='large',fontweight='bold')
plt.grid()
plt.show()


# In[34]:


lambbda = 5


# In[35]:


ExpDelay = EN/lambbda


# In[36]:


ExpDelay


# # 4(a) 

# In[37]:


def erlang(lammbda,uniform):
    U = uniform
    E = 0
    for i in range(4):
        E += (-1/lammbda)*np.log(U[i])
    return E


# In[38]:


plt.figure()
total = []
r2 = []
problist2 = []
for i in range(10000):
    E = erlang(24,uniform(0,1,4))
    r2.append(i)
    problist2.append(E)
    total.append(E)
plt.figure()
plt.plot(r2,problist2)
plt.title('Erlang Distribution',fontsize='large',fontweight='bold')
plt.grid()
plt.show()


# In[39]:


InterArrivalT = ExpRand(5,uniform(0,1,9999))
ServiceT = total
Arrival_0 = 0
Leave_0 = ServiceT[0]
ArrivalQueue = [[Arrival_0,1]]


# In[40]:


temp = 0
for i in range(len(InterArrivalT)):
    temp += InterArrivalT[i]
    ArrivalQueue.append([temp,1])
LeaveQueue = [[0 for col in range(2)] for row in range(10000)]
LeaveQueue[0] = [Leave_0,-1]


# In[41]:


for i in range(1,len(ArrivalQueue)):
    if ArrivalQueue[i][0] >= LeaveQueue[i-1][0]:
        LeaveQueue[i][0] = ArrivalQueue[i][0] + ServiceT[i]
        LeaveQueue[i][1] = -1
    else:
        LeaveQueue[i][0] = LeaveQueue[i-1][0] + ServiceT[i]
        LeaveQueue[i][1] = -1


# In[42]:


TimeLine = []
templist = []
templist.extend(ArrivalQueue)
templist.extend(LeaveQueue)
TimeLine = sorted(templist, key=lambda state: state[0])
state = TimeLine[0][1]


# In[43]:


for i in range(1,len(TimeLine)):
    state += TimeLine[i][1]
    TimeLine[i][1] = state


# In[44]:


maxstate = 0
for i in range(len(TimeLine)):
    if TimeLine[i][1] >= maxstate:
        maxstate = TimeLine[i][1]
maxstate


# In[45]:


def Pn(TimeLine):
    EN = 0
    r = []
    reslist = []
    for j in range(maxstate+1):
        timecount = 0
        for i in range(1,len(TimeLine)):
            if TimeLine[i-1][1] == j:
                timecount += (TimeLine[i][0] - TimeLine[i-1][0])
        res = timecount/TimeLine[-1][0]
        EN += j*res
        r.append(j)
        reslist.append(res)
    return EN,r,reslist


# In[46]:


EN,r3,reslist = Pn(TimeLine)


# In[47]:


EN


# In[48]:


plt.figure()
plt.grid()
plt.xlabel('state')
plt.ylabel('P(n)')
plt.title('P(n) of M/Ek/1',fontsize='large',fontweight='bold')
plt.plot(r3,reslist)
plt.show()


# # 4(c) 

# In[49]:


def erlang(lammbda,uniform):
    U = uniform
    E = 0
    for i in range(40):
        E += (-1/lammbda)*np.log(U[i])
    return E


# In[50]:


plt.figure()
ran = []
Elist = []
total = []
for i in range(10000):
    E = erlang(240,uniform(0,1,40))
    ran.append(i)
    Elist.append(E)
    total.append(E)
plt.grid()
plt.plot(ran,Elist)
plt.show()


# In[52]:


duniform = uniform(0,1,9999)
plt.figure()
mlist = []
ENlist = []
for m in np.arange(1,5.9,0.1):
    InterArrivalT = ExpRand(m,duniform)
    ServiceT = total
    Arrival_0 = 0
    Leave_0 = ServiceT[0]
    ArrivalQueue = [[Arrival_0,1]]

    temp = 0
    for i in range(len(InterArrivalT)):
        temp += InterArrivalT[i]
        ArrivalQueue.append([temp,1])
    LeaveQueue = [[0 for col in range(2)] for row in range(10000)]
    LeaveQueue[0] = [Leave_0,-1]

    for i in range(1,len(ArrivalQueue)):
        if ArrivalQueue[i][0] >= LeaveQueue[i-1][0]:
            LeaveQueue[i][0] = ArrivalQueue[i][0] + ServiceT[i]
            LeaveQueue[i][1] = -1
        else:
            LeaveQueue[i][0] = LeaveQueue[i-1][0] + ServiceT[i]
            LeaveQueue[i][1] = -1

    TimeLine = []
    templist = []
    templist.extend(ArrivalQueue)
    templist.extend(LeaveQueue)
    TimeLine = sorted(templist, key=lambda state: state[0])
    state = TimeLine[0][1]

    for i in range(1,len(TimeLine)):
        state += TimeLine[i][1]
        TimeLine[i][1] = state

    maxstate = 0
    for i in range(len(TimeLine)):
        if TimeLine[i][1] >= maxstate:
            maxstate = TimeLine[i][1]

    EN = 0
    
    for j in range(maxstate+1):
        timecount = 0
        for i in range(1,len(TimeLine)):
            if TimeLine[i-1][1] == j:
                timecount += (TimeLine[i][0] - TimeLine[i-1][0])
        res = timecount/TimeLine[-1][0]
        EN += j*res
    mlist.append(m)
    ENlist.append(EN)
plt.grid()    
plt.plot(mlist,ENlist,label = 'M/Ek/1',linestyle = '--',color='black',marker='>')
plt.legend(loc='upper right')




mlist = []
ENlist = []
for m in np.arange(1,5.9,0.1):
    InterArrivalT = ExpRand(m,duniform)
    ServiceT = [1/6 for i in range(10000)]
    Arrival_0 = 0
    Leave_0 = ServiceT[0]
    ArrivalQueue = [[Arrival_0,1]]

    temp = 0
    for i in range(len(InterArrivalT)):
        temp += InterArrivalT[i]
        ArrivalQueue.append([temp,1])
    LeaveQueue = [[0 for col in range(2)] for row in range(10000)]
    LeaveQueue[0] = [Leave_0,-1]

    for i in range(1,len(ArrivalQueue)):
        if ArrivalQueue[i][0] >= LeaveQueue[i-1][0]:
            LeaveQueue[i][0] = ArrivalQueue[i][0] + ServiceT[i]
            LeaveQueue[i][1] = -1
        else:
            LeaveQueue[i][0] = LeaveQueue[i-1][0] + ServiceT[i]
            LeaveQueue[i][1] = -1

    TimeLine = []
    templist = []
    templist.extend(ArrivalQueue)
    templist.extend(LeaveQueue)
    TimeLine = sorted(templist, key=lambda state: state[0])
    state = TimeLine[0][1]

    for i in range(1,len(TimeLine)):
        state += TimeLine[i][1]
        TimeLine[i][1] = state

    maxstate = 0
    for i in range(len(TimeLine)):
        if TimeLine[i][1] >= maxstate:
            maxstate = TimeLine[i][1]

    EN = 0
    
    for j in range(maxstate+1):
        timecount = 0
        for i in range(1,len(TimeLine)):
            if TimeLine[i-1][1] == j:
                timecount += (TimeLine[i][0] - TimeLine[i-1][0])
        res = timecount/TimeLine[-1][0]
        EN += j*res
    mlist.append(m)
    ENlist.append(EN)
plt.grid()    
plt.plot(mlist,ENlist,label='M/D/1',marker='*',color='red')
plt.legend(loc='upper right')
plt.grid()
plt.show()
# plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


#3


# In[ ]:





# In[ ]:





# In[ ]:




