import os
import numpy as np
import matplotlib.pyplot as plt
'''
mode 1 one side hoeffding
mode 2 two side hoeffding
mode 3 one side bernstein
mode 4 two side bernstein
mode 5 one side emp bernstein
mode 6 two side emp bernstein
'''


m = 10000
n = 100

delta = 0.1

mu = 0.5
variance  = 1.0 /12

epsilon = 0.005

modes =5

a  = np.random.rand(n*m)
a = a.reshape([m,n])


def cal_error(mu,x,variance=None,modes=1):
    sample_mu = np.mean(x)
    if modes ==1 or modes ==3 or modes ==5:
        return(sample_mu-mu)
    elif modes ==2 or modes ==4 or modes == 6:
        return(np.abs(sample_mu))

def cal_epsilon(delta, n ,variance=None,modes=1):
    if modes ==1:
        return(np.sqrt(np.log(1.0 / delta) / 2 / n ))
    elif modes ==2:
        return(np.sqrt(np.log(2.0 / delta) / 2 / n ))
    elif modes == 3:
        return(np.sqrt( 2* variance * np.log(1.0 / delta) /n ) + (np.log(1.0 / delta)) / (3.0*n))
    elif modes == 4:
        return( np.sqrt( 2* variance * np.log(2.0 / delta) /n ) + (np.log(2.0 / delta)) / (3.0*n))
    elif modes == 5 :
        return( np.sqrt( 2* variance * np.log(1.0 / delta) /n ) + (7*np.log(1.0 / delta)) / (3.0*(n-1)) )
    elif modes ==6:
        return(  np.sqrt( 2* variance * np.log(2.0 / delta) /n ) + (7*np.log(2.0 / delta)) / (3.0*(n-1))   )


def cal_delta(epsilon,n ,variance=None,modes=1):
    if modes ==1 :
        return(1.0*np.exp(-2*(epsilon**2) *n))
    elif modes ==2:
        return(2.0*np.exp(-2*(epsilon**2) *n))
    elif modes ==3:
        return( 1.0 * np.exp(-1.0* ( n*(epsilon**2)  )  / ( 2 * (variance**2)  + 2*epsilon /3 )) )
    elif modes ==4:
        return( 2.0 * np.exp(-1.0* ( n*(epsilon**2)  )  / ( 2 * (variance**2)  + 2*epsilon /3 )) )
    elif modes ==5:
        return (1.0 * np.exp(-1.0 * ((n-1)* (epsilon ** 2)) / (2 * (variance ** 2) + 14* epsilon / 3)))
    elif modes ==6:
        return (2.0 * np.exp(-1.0 * ((n-1)* (epsilon ** 2)) / (2 * (variance ** 2) + 14* epsilon / 3)))

def cal_n(epsilon,delta,variance=None , modes=1):
    if modes ==1 :
        return(np.log(1.0 / delta) / 2.0 /  epsilon / epsilon)
    elif modes ==2:
        return(np.log(2.0 / delta) / 2.0 /  epsilon / epsilon)
    elif modes ==3:
        return( np.log(1.0 / delta) *( 2 * (variance**2)  + 2*epsilon /3 ) / (epsilon**2))
    elif modes ==4:
        return (np.log(2.0 / delta) * (2 * (variance ** 2) + 2 * epsilon / 3) / (epsilon ** 2))
    elif modes ==5:
        return( np.log(1.0 / delta) *( 2 * (variance**2)  + 14*epsilon /3 ) / (epsilon**2))
    elif modes ==6:
        return( np.log(2.0 / delta) *( 2 * (variance**2)  + 14*epsilon /3 ) / (epsilon**2))


print('modes: ' ,modes)


if modes<=4:
    def run1():
        error  = np.array([cal_error(mu , x, variance = variance,modes=modes) for x  in a])
        epsilon_theory = cal_epsilon(delta , n , variance = variance,modes=modes)
        perc = np.percentile(error , (1-delta)*100)
        print('epsilon theory:' ,epsilon_theory , '    percentile:' , perc)



    def run2():
        error = np.array([cal_error(mu , x, variance = variance,modes=modes) for x  in a])
        delta = cal_delta(epsilon , n, variance = variance,modes=modes)
        delta_sample = sum(error > epsilon) *1.0/ m

        print( '1 - delta theory:' ,1 - delta, ' 1 - delta sample:' , 1  - delta_sample )


    #
    def run3():

        # delta = cal_delta(epsilon, n)
        n_theroy = cal_n(epsilon , delta, variance = variance,modes=modes)
        for ii in  range(2,2*n):
            this_a = a[:ii]
            # epsilon_theory = cal_epsilon(delta,ii)
            error = np.array([cal_error(mu, x, variance = variance,modes=modes) for x in this_a])
            perc = np.percentile(error , (1-delta)*100)
            if perc < epsilon :
                break
        print('n theory' , n_theroy ,'n_sample',ii )


else:
    def run1():
        ##given delta n
        error  = np.array([cal_error(mu , x, variance = np.var(x),modes=modes) for x  in a])
        epsilon_theory = np.array([cal_epsilon(delta , n , variance = np.var(x),modes=modes) for x in a])
        # perc = np.percentile(error , (1-delta)*100)
        # print('epsilon theory:' ,epsilon_theory , '    percentile:' , perc)
        print(sum(error<epsilon_theory) / m)

    def run2():
        ##given epsilon n
        # error = np.array([cal_error(mu , x, variance = np.var(x),modes=modes) for x  in a])
        # delta = cal_delta(epsilon , n, variance = variance,modes=modes)
        # delta_sample = sum(error > epsilon) *1.0/ m
        #
        # print( '1 - delta theory:' ,1 - delta, ' 1 - delta sample:' , 1  - delta_sample )
        pass


    #
    def run3():
        ## given delta epsilon
        all_a = a[0,:]


        for ii in  range(2,2*n):
            this_a = all_a[:ii]
            n_theroy = cal_n(epsilon, delta, variance=np.var(this_a), modes=modes)

            # error = np.array([cal_error(mu, x, variance = np.var(x),modes=modes) for x in this_a])
            # perc = np.percentile(error , (1-delta)*100)
            # if perc < epsilon :
            if ii > n_theroy:
                break

        print('n theory' , n_theroy ,'n_sample',ii )


print('result of run 1')
run1()
print('result of run 2')
run2()
print('result of run 3')
run3()

