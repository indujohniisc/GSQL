
# coding: utf-8

# In[ ]:


import numpy as np
import mdptoolbox, mdptoolbox.example
import random
import sys
#import matplotlib.pyplot as plt
#from numpy.random import seed



#np.random.seed(0)
#P, R = mdptoolbox.example.forest()
s = 10
a= 5
discount = 0.6
q_count = 0
sor_q_count = 0
sql_q_count = 0
gsql_q_count = 0
gsql1_q_count = 0
policy_count = 0
sor_policy_count = 0
sql_policy_count = 0
gsql_policy_count = 0
gsql1_policy_count = 0
percentage_count = 0

episodes = 100
iterations = 100000

normal_total_diff = np.zeros((episodes,iterations))
#sor_total_diff= np.zeros((episodes,iterations))
sql_total_diff= np.zeros((episodes,iterations))
gsql1_total_diff= np.zeros((episodes,iterations))
gsql2_total_diff=np.zeros((episodes,iterations))
#gsql_total_diff= np.zeros((5,episodes,iterations))

#w_list=np.array([0.6,0.8,1,1.2,1.4])

for count in range(episodes):
    print(count)
    np.random.seed((count+1)*100)
    random.seed((count+1)*110)
    
    P, R = mdptoolbox.example.rand(s, a)
    #print(P)
    #print(np.min(P))

    vi = mdptoolbox.mdp.ValueIteration(P, R, discount,epsilon=0.00001)
    vi.run()
    #print(vi.policy)

    #print('************************************')
    ql = mdptoolbox.mdp.QLearning(P, R, discount,n_iter=iterations)
    ql.setVerbose()
    ql.run()
    #print(ql.V)
    #print(np.shape(vi.V) )
    #print(ql.policy,np.linalg.norm((np.asarray(vi.V) - np.asarray(ql.V))))
    #print('************************************')
    #ql2 = mdptoolbox.mdp.SOR_QLearning(P, R, discount,n_iter=iterations)
    #ql2.run()
    #Speedy QL
    ql3 = mdptoolbox.mdp.SpeedyQLearning(P, R, discount,n_iter=iterations)
    ql3.run()
    norm_diff_sql = vi.V - ql3.q_values
    sql_total_diff[count] = np.linalg.norm(norm_diff_sql,axis =1)
    #print(ql3.w)
    #print(ql3.q_values)
    #print(ql3.q_values1)
    #GSQL
    #if(ql3.w>1.3):
        #print("Got w>1.3")
    #for i in range(len(w_list)) :
    ql4 = mdptoolbox.mdp.GSQL(P, R, discount,n_iter=iterations)
    ql4.run()
    norm_diff_gsql1 = vi.V - ql4.q_values
    gsql1_total_diff[count] = np.linalg.norm(norm_diff_gsql1,axis =1)
    #GSQL1
    ql5 = mdptoolbox.mdp.GSQL2(P, R, discount,n_iter=iterations)
    ql5.run()
    norm_diff_gsql2 = vi.V - ql5.q_values
    gsql2_total_diff[count] = np.linalg.norm(norm_diff_gsql2,axis =1)
    #print(np.linalg.norm((np.asarray(ql2.V) - np.asarray(ql3.V))))
    #print(ql2.V)
    #q_count += np.linalg.norm((np.asarray(vi.V) - np.asarray(ql.V)))
    #sor_q_count +=np.linalg.norm((np.asarray(vi.V) - np.asarray(ql2.V)))
    #sql_q_count +=np.linalg.norm((np.asarray(vi.V) - np.asarray(ql3.V)))
    #gsql_q_count +=np.linalg.norm((np.asarray(vi.V) - np.asarray(ql4.V)))
    #gsql1_q_count +=np.linalg.norm((np.asarray(vi.V) - np.asarray(ql5.V)))
    
    #policy_count += np.sum(vi.policy != ql.policy)
    #sor_policy_count += np.sum(vi.policy != ql2.policy)
    #sql_policy_count += np.sum(vi.policy != ql3.policy)
    #gsql_policy_count += np.sum(vi.policy != ql4.policy)
    #gsql1_policy_count += np.sum(vi.policy != ql5.policy)
    
#     if np.linalg.norm((np.asarray(vi.V) - np.asarray(ql2.V))) < np.linalg.norm((np.asarray(vi.V) - np.asarray(ql.V))):
#         percentage_count = percentage_count + 1
    #print(np.shape(vi.V),np.shape(ql.q_values))
    #norm_diff = vi.V - ql.q_values
    #print(np.shape(norm_diff))
    #normal_total_diff[count] = np.linalg.norm(norm_diff,axis =1)
    
    #norm_diff_sor = vi.V - ql2.q_values
    #sor_total_diff[count] = np.linalg.norm(norm_diff_sor,axis =1)
    
    #norm_diff_sql = vi.V - ql3.q_values
    #sql_total_diff[count] = np.linalg.norm(norm_diff_sql,axis =1)
    
    #norm_diff_gsql = vi.V - ql4.q_values
    #gsql_total_diff[count] = np.linalg.norm(norm_diff_gsql,axis =1)
    
    #norm_diff_gsql1 = vi.V - ql5.q_values
    #gsql1_total_diff[count] = np.linalg.norm(norm_diff_gsql1,axis =1)

    #norm_diff1 = ql.q_values-ql3.q_values
    #print(norm_diff1)
#print(np.shape(normal_total_diff))
#avg_error_ql=np.mean(normal_total_diff,axis=0)
#print(np.shape(avg_error_ql))
#avg_error_sor=np.mean(sor_total_diff,axis=0)
avg_error_sql=np.mean(sql_total_diff,axis=0)
std_error_sql=np.std(sql_total_diff,axis=0)
avg_error_gsql1=np.mean(gsql1_total_diff,axis=0)
std_error_gsql1=np.std(gsql1_total_diff,axis=0)
avg_error_gsql2=np.mean(gsql2_total_diff,axis=0)
std_error_gsql2=np.std(gsql2_total_diff,axis=0)
#print(np.linalg.norm(avg_error_ql-avg_error_sql))
#avg_error_gsql=np.zeros((len(w_list),iterations))
#for i in range(len(w_list)):
    #print(np.shape(gsql_total_diff[i]))
#    avg_error_gsql[i]=np.mean(gsql_total_diff[i],axis=0)
#np.savetxt("w_gsql2.csv", np.c_[avg_error_gsql[0],avg_error_gsql[1],avg_error_gsql[2],avg_error_gsql[3],avg_error_gsql[4]], delimiter=",") 
#
np.savetxt("mean_std.csv", np.c_[avg_error_sql,avg_error_gsql1,avg_error_gsql2,std_error_sql,std_error_gsql1,std_error_gsql2], delimiter=",")    

#np.savetxt("plot_values.csv", np.c_[avg_error_ql,avg_error_sor,avg_error_sql,avg_error_gsql], delimiter=",")
#    
   # print("SOR comparison")
   # print(np.linalg.norm((np.asarray(vi.V) - np.asarray(ql.V))),np.linalg.norm((np.asarray(vi.V) - np.asarray(ql2.V))),np.sum(vi.policy != ql.policy),np.sum(vi.policy != ql2.policy),ql2.w)
    #print(np.linalg.norm((np.asarray(vi.V) - np.asarray(ql2.V))),np.sum(vi.policy != ql2.policy),ql2.w)
   # print("\nSQL comparison")
    #print(np.linalg.norm((np.asarray(vi.V) - np.asarray(ql.V))),np.linalg.norm((np.asarray(vi.V) - np.asarray(ql3.V))),np.sum(vi.policy != ql.policy),np.sum(vi.policy != ql3.policy),ql3.w)
#print("QL_count=",q_count)
#print("\nSOR_QL_count=",sor_q_count)
#print("\nSQL_count=",sql_q_count)
#print("\nGSQL_count=",gsql_q_count)
#
#print("QL_count=",policy_count)
#print("\nSOR_QL_count=",sor_policy_count)
#print("SQL_count=",sql_policy_count)

