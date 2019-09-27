import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad
import sys

#question 1 is math...I thought it would take less time than it is.


##########problem 2
#linear interpolation b/c I am lazy

def interp_volt(volt = ((np.random.random_sample() * (1.644290-0.090681)) + 0.090681)):
	#linearly interpolate a voltage value to get a corresponding temperature estimate
	#(if no input voltage, randomly generate one within the bounds of the file)
	if volt >= 1.644290 or volt <= 0.090681:
		print "fuck off, man"
		return

	#load in temperature and voltage values from text file
	textin = np.loadtxt('lakeshore.txt')
	voltage = textin[:,1]
	temperature = textin[:,0]

	#find nearest values in voltage array to voltage we want
	diff = np.abs(voltage - volt)
	indy = diff.argmin()
	if diff[indy+1] <= diff[indy-1]: indy_2 = indy+1 
	else: indy_2 = indy-1

	#calculate slope and y intercept of linear fit
	m = (temperature[indy_2]-temperature[indy])/(voltage[indy_2]-voltage[indy])
	b = temperature[indy] - m*voltage[indy]

	#calculate fit value with error
	interp_temp = m*volt + b
	interp_error = ((voltage[indy] - voltage[indy_2])**2.)/8.

	#return temperature, error in K
	return interp_temp, interp_error

print interp_volt()


##########problem 3

def integrate_dumb(func, a, b, tol=1e-8, pass_x=None, x_glob=None, y_glob=None):
	#define x, y values
	x=np.linspace(a,b,5)
	dx = (b-a)/4.
	y = np.zeros(5)
	#yeah, pass_x ended up just being a recursion flag.. I could rename it but why bother?
	if not np.any(pass_x):
		y[:] = func(x)
		n_eval = 5
		#set up arrays to hold already calculated x and y values
		x_glob = x
		y_glob = y
	else:
		#if there are x values that have already been evaluated, grab them
		if np.any(np.isin(x, x_glob)):
			#could assume based on the fact that we're using 5 points that there are three that carry over every time, but where's the fun in that??? let's find them every time instead!
			indices_x_exist = np.where(np.isin(x, x_glob)==True)[0]
			for i in indices_x_exist:
				y[i] = y_glob[np.where(x_glob == x[i])[0]]
			extra = np.arange(0,5)
			indices_x_nonexist = np.setdiff1d(extra, indices_x_exist)
			#only evaluate points we actually need
			y[indices_x_nonexist] = func(x[indices_x_nonexist])
			#add newly evaluated points to array of values we've already calculated
			x_glob = np.append(x_glob, x[indices_x_nonexist])
			y_glob = np.append(y_glob, y[indices_x_nonexist])
			n_eval = 5-len(indices_x_exist)
#			print "a"
#			print indices_x_exist, " | ", indices_x_nonexist
#			print x_glob
		else:
			#is this ever actually used? I don't think so...but let's leave it just in case
			y[:] = func(x)
			n_eval = 5
			x_glob = np.append(x_glob, x)
			y_glob = np.append(y_glob, y)	
			print "I'm useful!"

	f_dumber = (b-a)*(y[0]+4.*y[2]+y[4])/6.
	f_dumb = (b-a)*(y[0]+4.*y[1]+2.*y[2]+4.*y[3]+y[4])/12.
	error = np.abs(f_dumb-f_dumber)
	if error < tol:
		return (16.*f_dumb-f_dumber)/15., error, n_eval
	else:
		midpt = 0.5*(b+a)
		f_left,error_left,eval_left = integrate_dumb(func, a, midpt, tol/2., pass_x = x, x_glob=x_glob, y_glob=y_glob)
		f_right,error_right,eval_right = integrate_dumb(func, midpt, b, tol/2., pass_x = x, x_glob=x_glob, y_glob=y_glob)
		f = f_left + f_right
		n_eval = n_eval+eval_left+eval_right
		error = error_left+error_right
		return f, error, n_eval

def integrate_dumber(func, a, b, tol=1e-8):
	#ripped from prof
	x=np.linspace(a,b,5)
	dx = (b-a)/4.
	y=func(x)
	n_eval = x.size
	f_dumber = (y[0]+4.*y[2]+y[4])/6.*(b-a)
	f_dumb = (y[0]+4.*y[1]+2.*y[2]+4.*y[3]+y[4])/12.*(b-a)
	error = np.abs(f_dumb-f_dumber)
	if error < tol:
		return (16.*f_dumb-f_dumber)/15., error, n_eval
	else:
		midpt = 0.5*(b+a)
		f_left,error_left,eval_left = integrate_dumber(func, a, midpt, tol/2.)
		f_right,error_right,eval_right = integrate_dumber(func, midpt, b, tol/2.)
		f = f_left + f_right
		n_eval = n_eval+eval_left+eval_right
		error = error_left+error_right
		return f, error, n_eval

#saves a couple hundred function calls or so
f, err, neval = integrate_dumb(np.exp, -1, 1, 1e-8);pred=np.exp(1)-np.exp(-1)
print('f,err,neval are ' + repr([f,err,neval])+' with err ' + repr(np.abs(f-pred)))
f, err, neval = integrate_dumber(np.exp, -1, 1, 1e-8);pred=np.exp(1)-np.exp(-1)
print('f,err,neval are ' + repr([f,err,neval])+' with err ' + repr(np.abs(f-pred)))



##########problem 4

def chg(theta):
	#say r=1 for convenience
	return ((z-np.cos(theta))*np.sin(theta))/((1.+(z**2.)-(2.*z*np.cos(theta)))**1.5)

result=np.array([])
result_other=np.array([])

#for easy changing step size & plotting convenience
xplt = np.arange(0,3,0.1)

for i in xplt:
	z = i #just for my own sanity
	f, err, neval = integrate_dumb(chg, 1e-5, np.pi)
	f_other, err = quad(chg, 1e-5, np.pi)
	result = np.append(result, f)
	result_other = np.append(result_other, f_other)

#cool the integral works in both functions! no singularity! (though for some reason some step sizes make it blow up? not really sure why)

plt.figure(1)
plt.plot(xplt,result)
plt.title('my integrator')
plt.xlabel('z (R=1)')
plt.ylabel('charge')

plt.figure(2)
plt.plot(xplt,result_other)
plt.title('python integrator')
plt.xlabel('z (R=1)')
plt.ylabel('charge')
plt.show()






