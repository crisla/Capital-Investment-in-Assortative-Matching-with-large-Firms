"""
 Python functions for solving and plotting the solution to
 an Assortative Matching with Large Firms (2016) model with capital
 via shooting

@author : Cristina Lafuente
@date : 11/02/2020
"""

# Generic modules
# from __future__ import division
import numpy as np
import sympy as sym
from numpy.polynomial.polynomial import polyval
from scipy.optimize import least_squares as leastsq
from scipy.interpolate import interp1d
import operator
from scipy.special import erf
# Plotting modules
import matplotlib.pyplot as plt
import seaborn as sn
sn.set_style('whitegrid')
cp = sn.xkcd_rgb["pinkish"]
cb = "#3498db"
cr = "#1fa774"
# Specific modules
import inputs
import models
import shooting
from numba import njit, prange

_modules = [{'ImmutableMatrix': np.array, 'erf': erf, 'sqrt': np.sqrt}, 'numpy']

x, y = sym.var('x, y')
R, l, r, rho, gamma, eta, L, A, kappa, p = sym.var('R, l, r, rho, gamma, eta, L, A, kappa, p')
x0,x1= [3.16e-6,1e2]

# PART 0 : AUXILIARY FUNCTIONS
@njit
def hyper_root_1(p,q):
	# CASE 1: In case p<0 and 4p^3+27q^2>0 we have one root:
	return -2 * (np.abs(q)/q) * np.sqrt((-p/3)) * np.cosh( (1/3)*np.arccosh( ((-3*abs(q))/(2*p)) * np.sqrt((-3/p)) ) )

@njit
def hyper_root_2(p,q):
	# CASE 2: In case p>0, we also have one root:
	return -2* (p/3)**(0.5) * np.sinh( (1/3)*np.arcsinh( ((3*q)/(2*p)) * (3/p)**(0.5) ) )

@njit
def hyper_root_3(p,q):
	# CASE 3: In case p<0 and 4p^3+27q^2<=0 we have three real roots, of which the biggest is:
	return 2*np.sqrt((-p/3))*np.cos( (1/3)*np.arccos( ((3*q)/(2*p)) * np.sqrt((-3/p)) ) )


@njit(parallel=True)
def jitted_trig_condition_pq(p,q):
	'''
	Condition for three real roots or 1 real root.
	
	Input: 
	p 	(float): value of the p part of the solution (refer to readme)
	q 	(float): value of the q part of the solution (refer to readme)
	
	Output: An interger for the case in which we are in (see above)
	'''
	pq = 4*p**3+27*q**2
	if p<0 and pq<=0:
		return 3
	elif p<0 and pq>0:
		return 1
	elif p>0:
		return 2

@njit(parallel=True)
def jitted_k_star_pq(p,q):
	'''
	Function that returns the trigonometric solution for capital.

	Notice that the production function has been inverted (x is y, y is x, l is 1/l)
	
	Input: 
	p  (float): value of the p part of the solution (refer to readme)
	q  (float): value of the q part of the solution (refer to readme)
	
	Output: Optimal capital (note that is the true capital, that is, raised to the 4th power)
	'''
	
	if jitted_trig_condition_pq(p,q) == 1:
		return hyper_root_1(p,q)**4
	elif jitted_trig_condition_pq(p,q) == 3:
		return hyper_root_3(p,q)**4
	else:
		return hyper_root_2(p,q)**4


@njit(parallel=True)
def global_k_star(xgrid,ygrid,lgrid,params,ftype=0):
	n = xgrid.size
	nx = ygrid.size
	nl = lgrid.size
	
	A, eta, kappa, R = params
	
	k_sol = np.zeros((n,nx,nl))
	
	# Please note the inversion of x and y
	
	for i in prange(n):
		yi = xgrid[i]
		for k in prange(nx):
			xk = ygrid[k]
			for j in prange(nl):
				lj = lgrid[j]
				if ftype==0:
					p = -(eta**2*A*kappa)/(2*R)
					q = -(eta*(1-eta)*kappa*A*(xk*(1/lj)**yi)**(1/4))/(2*R)
				# With A
				elif ftype==1:
					p = -(eta**2*A*kappa*yi)/(2*R)
					q = -(eta*(1-eta)*kappa*A*yi*(xk*(1/lj))**(1/4))/(2*R)
				# (xk)^(1/4)
				elif ftype==2:
					p = -(eta**2*A*kappa*yi**(1/2))/(2*R) 
					q = -(eta*(1-eta)*kappa*A*(xk*(1/lj)*yi)**(1/4))/(2*R)
				# x(k)^(1/4)
				else:
					p = -(eta**2*A*kappa*yi**2)/(2*R) 
					q = -(eta*(1-eta)*kappa*A*yi*(xk*(1/lj))**(1/4))/(2*R)
				k_sol[i,k,j] = jitted_k_star_pq(p,q)
	
	return k_sol

@njit
def residuals_k_star(params, kr, a, b, z):
	"""
	Returns value of a polynomial of the 3rd degree in three dimensions (a,b,z) minus the value of capital (kr)

	It is assumed that labour and skill are inputed in logs, land quality in levels.
	"""
	a0, a1,a2,b1,b2,ab,a2b,ab2,a2b2,a3,b3,a3b,ab3, a3b2, a2b3, a3b3, z1, z1a, z1a, z1b, z1a2, z1b2, z2a,z2b,z2a2,z2b2, z1ab, z1a2b,z1ab2, z1a2b2, z2ab, z2a2b, z2ab2, z2a2b2, z3, z3a, z3b, z3a2, z3b2, z3a3, z3b3, z3ab, z3a2b, z3ab2, z3a2b2, z3a3b, z3ab3, z3a3b2, z3a2b3, z3a3b3 = params
	return a0 + a1*a + a2*a**2 + b1*b + b2*b**2 + ab*a*b + a2b*a**2*b + ab2*a*b**2 + a2b2*a**2*b**2 + \
			a3* a**3 + b3*b**3 + a3b*a**3*b + ab3*a*b**3 + a3b2*a**3*b**2 + a2b3*a**2*b**3 + a3b3* a**3*b**3 + \
			z1*z + z1a*z*a + z1b*z*b + z1a2*z*a**2 + z1b2*z*b**2 + z2a*z**2*a + z2b*z**2*b + z2a2*z**2*a**2 + \
			z2b2*z**2*b**2 + z1ab*z*a*b + z1a2b*z*a**2*b + z1ab2*z*a*b**2 + z1a2b2*z*a**2*b**2 + \
			z2ab*z**2*a*b + z2a2b*z**2*a**2*b + z2ab2*z**2*a*b**2 + z2a2b2*z**2*a**2*b**2 + \
			z3*z**3 + z3a*z**3*a + z3b*z**3*b + z3a2*z**3*a**2 + z3b2*z**3*b**2 + z3a3*z**3*a**3 + \
			z3b3*z**3*b**3 + z3ab*z**3*a*b + z3a2b*z**3*a**2*b + z3ab2*z**3*a*b**2 + z3a2b2*z**3*a**2*b**2 + \
			z3a3b*z**3*a**3*b + z3ab3*z**3*a*b**3 + z3a3b2*z**3*a**3*b**2 + z3a2b3*z**3*a**2*b**3 + \
			z3a3b3*z**3*a**3*b**3 - kr


# PART I : SOLVING FUNCTIONS

def get_k(params, ftype=0, plot=False, save_in=None, x_range=[0.999,1.001], nx=20, ny=100, nl=100, print_residuals=False):
	"""
	Calcualtes three approximated functions of k(x,y,l,r) and returns the one with the lowest MSE.

	WARNING: get_k() (and the solver) use an inverted production function, such that:
		* output xs are model ys
		* output ys and model xs and
		* output thetas are model 1/l
		
	So the output k is actually defined as k(y,x,1/l,1/r)	

	Inputs
	------
	params	: (dic) Dictionary with country parameters. Must contain [rho, gamma, eta, A, kappa, p]
	ftype 	: (int) Type of function. Possible values:
					* 0 : x as an exponent of (l/r)
					* 1 : x multiplying k
					* 2 : x as a scaling factor (multiplying A)
					* 3 : x multiplying k (but not raised to rho)
	fchoice	: (int) Type of approximation function to k. If None, choses best fit measured by MSE.
					Values:
					* 0 for cuadratic tensors
					* 1 for cubic tensors
					* 2 for polynomial on x(r/l)^y (WARNING: not well behaved)

	plot 	: (bool) True for producing a plot of the fit of the three kinds
	save_in : (bool) True for saving the paramers. File names will be:
					- "pams_new.csv" -> Parameters of the polynomial on (x(r/l)^y)
					- "pams3.csv" -> Parameters of the cubic tensors
					- "pams.csv" -> Parameters of the quadratic tensors
	x_range : (list) List or tuple with the somaing for the spread of x. Default is [0.999,1.001].
	
	nx		: (int)  Number of gridpoints for the spread of x. Default is 20.
	ny		: (int)  Number of gridpoints for the spread of y. Default is 100.
	nl		: (int)  Number of gridpoints for the spread of l. Default is 100.

	print_residuals : (bool) If true, prints the average square resudial of the approximation. 
							That is: mean((k_approx-k)/k)^2)

	Output
	------
	k_star	: (sym) Symbolical expression for k(x,y,l,r)

	"""
	xg = np.linspace(x_range[0],x_range[1],nx)
	yg = np.exp(np.linspace(-10.0,np.log(100),ny))
	if params ['A']==1: # rich country case
		lg = 1/np.exp(np.linspace(1e-2,10.0,nl))
	else: # poor country case
		lg = 1/np.exp(np.linspace(-15.0,7.0,nl))
	

	# TO DO's: check the parameters are correct, the grid and bounds valid
	params_array = np.array((params ['A'],params ['eta'],params ['kappa'],params ['R']))	

	xgrid, ygrid, lgrid = np.meshgrid(xg,yg,lg, indexing='ij')

	k_star_grid = global_k_star(xg,yg,lg,params_array,ftype=ftype)
	
	
	# Residuals for 3rd degree polinomial
	coef_capital = leastsq(residuals_k_star,  np.zeros(50), args=(k_star_grid.ravel(), np.log(ygrid).ravel(), np.log(lgrid).ravel(), xgrid.ravel()),
						  method='lm')
	if print_residuals:
		print(np.mean((residuals_k_star(coef_capital.x,k_star_grid.ravel(), np.log(ygrid).ravel(), np.log(lgrid).ravel(), xgrid.ravel())/ k_star_grid.ravel())**2))

	# Unpacking coefficients
	a0, a1,a2,b1,b2,ab,a2b,ab2,a2b2,a3,b3,a3b,ab3, a3b2, a2b3, a3b3, z1, z1a, z1a, z1b, z1a2, z1b2, z2a,z2b,z2a2,z2b2, z1ab, z1a2b,z1ab2, z1a2b2, z2ab, z2a2b, z2ab2, z2a2b2, z3, z3a, z3b, z3a2, z3b2, z3a3, z3b3, z3ab, z3a2b, z3ab2, z3a2b2, z3a3b, z3ab3, z3a3b2, z3a2b3, z3a3b3 = coef_capital.x
	# Creating symbolical variables
	x, y, l = sym.var('x,y,l')
	# Symbolic expression for k_star
	k_star = a0 + a1*sym.log(x) + a2*sym.log(x)**2 + b1*sym.log(l) + b2*sym.log(l)**2 + ab*sym.log(x)*sym.log(l) + a2b*sym.log(x)**2*sym.log(l) + ab2*sym.log(x)*sym.log(l)**2 + a2b2*sym.log(x)**2*sym.log(l)**2 + \
	a3*sym.log(x)**3 + b3*sym.log(l)**3 + a3b*sym.log(x)**3*sym.log(l) + ab3*sym.log(x)*sym.log(l)**3 + a3b2*sym.log(x)**3*sym.log(l)**2 + a2b3*sym.log(x)**2*sym.log(l)**3 + a3b3*sym.log(x)**3*sym.log(l)**3 + \
	z1*y + z1a*y*sym.log(x) + z1b*y*sym.log(l) + z1a2*y*sym.log(x)**2 + z1b2*y*sym.log(l)**2 + z2a*y**2*sym.log(x) + z2b*y**2*sym.log(l) + z2a2*y**2*sym.log(x)**2 + \
	z2b2*y**2*sym.log(l)**2 + z1ab*y*sym.log(x)*sym.log(l) + z1a2b*y*sym.log(x)**2*sym.log(l) + z1ab2*y*sym.log(x)*sym.log(l)**2 + z1a2b2*y*sym.log(x)**2*sym.log(l)**2 + \
	z2ab*y**2*sym.log(x)*sym.log(l) + z2a2b*y**2*sym.log(x)**2*sym.log(l) + z2ab2*y**2*sym.log(x)*sym.log(l)**2 + z2a2b2*y**2*sym.log(x)**2*sym.log(l)**2 + \
	z3*y**3 + z3a*y**3*sym.log(x) + z3b*y**3*sym.log(l) + z3a2*y**3*sym.log(x)**2 + z3b2*y**3*sym.log(l)**2 + z3a3*y**3*sym.log(x)**3 + \
	z3b3*y**3*sym.log(l)**3 + z3ab*y**3*sym.log(x)*sym.log(l) + z3a2b*y**3*sym.log(x)**2*sym.log(l) + z3ab2*y**3*sym.log(x)*sym.log(l)**2 + z3a2b2*y**3*sym.log(x)**2*sym.log(l)**2 + \
	z3a3b*y**3*sym.log(x)**3*sym.log(l) + z3ab3*y**3*sym.log(x)*sym.log(l)**3 + z3a3b2*y**3*sym.log(x)**3*sym.log(l)**2 + z3a2b3*y**3*sym.log(x)**2*sym.log(l)**3 + \
	z3a3b3*y**3*sym.log(x)**3*sym.log(l)**3
	
	return k_star



def solve(country, k_sol, ftype=0, logs=True, spread='even',dispams=None,x_range=[0.99999,1.00001],
	assort="positive", verbose=False,integrator='lsoda',guess_overide=1.0,scaling_x=1.0,normconst=0.0):
	"""

	Solves a 2-sided heterogeneity matching model usign a shooting algorithm, given capital choice k.

	WARNING: the solver (and get_k()) use an inverted production function, such that:
		* output xs are model ys
		* output ys and model xs and
		* output thetas are model 1/l 

	Inputs
	------
	country (str)			: "R" or "P" for Rich or Poor country parameters
	k_sol (sympy expression):

	Optional
	........

	ftype (int)				: Specification of production function (default=0) (see "Introduction" in the notebook)
	logs (bool)				: True (default) for log steps.
	spread (str)			: 'even' (default) for uniform distribution of x
							  'lognormal' for lognormal distribution of x
							  'normal' for normal distribution of x
							  'bimod' for bimodal distribution of x (2 join normals with same variance and different means)
	disparams (list or tup) : in case that the spread is not even, refers to values of the mean and std of the distribtuions;
							  for the bimodal, it must have length=3 and contains mean1, mean2 and the common variance;
							  all distribtuions have defauls parameters if the user does not specify hers.
	x_range (list or tup)	: range for x
	assort (str)			: if "positive" (default) assumes (and forces) PAM, "negative" does NAM
	verbose	(bool)			: if True prints progress of the solver; False is default.
	guess_overide (float)	: if the user wants to start with a particular inital guess for maximum land size. Default is 1.
	scaling_x (float)		: scalar that multiplies the cdf of x, used to make sure it integrates to 1.
	normconst (float)		: scalar to noramlize the cdf of x, used to ensure cdf_x(max(x))=1

	Outputs
	-------
	A solution dictionary with he keys:
	"solved" (bool)					: True is the solver was successful
	"xs","ys","thetas",
	"ws","pis","ks" (numpy.arrays)	: arrays with the points along the solution path 
	"solution": (pandas.dataframe)	: all of the above in a dataframe for convinience
	"pdf_dis" (numpy.array)			: the discretized values of the pdf(y)
	"F" (sympy expression)			: Production function chosen above. Convient for calculating derivatives (below)
	"PAM" (str)						: Assortativity ('positive' or 'negative')

	"""
	x0,x1= [3.16e-6,1e2]
	y0,y1=x_range

	if country=="R":
		F_params = {'R':0.13099, 'rho':0.25, 'gamma':0.5, 'eta':0.890204456766942, 'A': 1.0, 'kappa':1.0, 'p':0.3159}
		x_scale = 169.249 * scaling_x
		k_star = k_sol
		guess = guess_overide
	else:
		F_params = {'R':0.3958, 'rho':0.25, 'gamma':0.5, 'eta':0.890204456766942, 'A': 0.3987, 'kappa':1.0, 'p':0.5209}
		x_scale = 19.595 * scaling_x
		k_star = k_sol
		guess = guess_overide*3


	# Firms
	if spread=='even':
		y, a, b = sym.var('y, a, b')
		skill_cdf = ((y - a) / (b - a)) * x_scale
		skill_params = {'a': y0, 'b': y1}
		skill_bounds = [y0, y1]
	elif spread=='lognorm':
		y, loc1, mu1, sigma1 = sym.var('y, loc1, mu1, sigma1')
		skill_cdf = (0.5 + 0.5 * sym.erf((sym.log(y - loc1) - mu1) / sym.sqrt(2 * sigma1**2)))* x_scale - normconst
		if dispams==None:
			skill_params = {'loc1': 0.0, 'mu1': -0.03, 'sigma1': 0.055}
		else:
			skill_params = {'loc1': 0.0, 'mu1': dispams[0], 'sigma1': dispams[1]}
		skill_bounds = [y0, y1]
	elif spread=='norm':
		sig1, y, mu1 = sym.var("sig1, y, mu1")
		skill_cdf = 0.5 * (1 + sym.erf((y-mu1)/ sym.sqrt(2 * sig1**2))) * x_scale - normconst
		if dispams==None:
			skill_params = {'mu1': 1.1, 'sig1': 0.05 }
		else:
			skill_params = {'mu1': dispams[0], 'sig1': dispams[1]}
		skill_bounds = [y0, y1]
	elif spread=='bimod':
		sig1, sig2, y, mu1, mu2= sym.var("sig1, sig2, y, mu1, mu2")
		c1 = 0.5 * (1 + sym.erf((y-mu1)/ sym.sqrt(2 * sig1**2)))
		c2 = 0.5 * (1 + sym.erf((y-mu2)/ sym.sqrt(2 * sig2**2)))
		skill_cdf  = (c1*0.5+c2*0.5) * x_scale - normconst
		if dispams==None:
			skill_params = {'mu1': 0.9, 'mu2': 1.1, 'sig1': 0.05,'sig2': 0.05}
		else:
			skill_params = {'mu1': dispams[0], 'mu2': dispams[1], 'sig1': dispams[2],'sig2': dispams[2]}
		skill_bounds = [y0, y1]
		
	else:
		raise ValueError('Unvalid spread. Choose even or lognorm.')

	firms = inputs.Input(var=y,cdf=skill_cdf,params=skill_params,bounds=skill_bounds)

	# Workers
	x, loc2, mu2, sigma2 = sym.var('x, loc2, mu2, sigma2')
	productivity_cdf = (0.5 + 0.5 * sym.erf((sym.log(x - loc2) - mu2) / sym.sqrt(2 * sigma2**2)))
	productivity_params = {'loc2': 0.0, 'mu2': -1.8316, 'sigma2': 4.6553}
	productivity_bounds = (x0, x1)

	workers = inputs.Input(var=x,cdf=productivity_cdf,params=productivity_params,bounds=productivity_bounds)

	# Production Function
	if ftype==0:
		F = (l)*p*A*kappa*(eta*(k_star)**rho + (1- eta)*(((r/l)**y)*x)**rho)**(gamma/rho) - (l)*R*k_star
	elif ftype==1: # xA
		F = y*(l)*p*A*kappa*(eta*(k_star)**rho + (1- eta)*((r/l)*x)**rho)**(gamma/rho) - (l)*R*k_star
	elif ftype==2: # xk
		F = (l)*p*A*kappa*(eta*(k_star*y)**rho + (1- eta)*((r/l)*x)**rho)**(gamma/rho) - (l)*R*k_star
	elif ftype==3: # x(k^1/4)
		F = (l)*p*A*kappa*(eta*y*(k_star)**rho + (1- eta)*((r/l)*x)**rho)**(gamma/rho) - (l)*R*k_star
	elif ftype==4: # k^x
		F = (l)*p*A*kappa*(eta*(k_star**y)**rho + (1- eta)*((r/l)*x)**rho)**(gamma/rho) - (l)*R*k_star

	#Setting up the solver
	model = models.Model(assortativity=assort,workers=workers,firms=firms,production=F,params=F_params)
	solver = shooting.ShootingSolver(model=model)

	# log steps
	if logs:
		nk = None
		kts = np.logspace(np.log10(x0),np.log10(x1),num=6000)
	# Uniform steps
	elif not logs:
		nk =6000
		kts = None

	# Solve
	solver.solve(guess, tol=1e-9, number_knots=nk, knots=kts,integrator=integrator, message=verbose)
	# try:
	# 	solver.solve(guess, tol=1e-9, number_knots=nk, knots=kts,integrator=integrator, message=verbose)
	# except AssertionError as e1:
	# 	print(e1)
	# 	try:
	# 		solver.solve(guess*10, tol=1e-9, number_knots=nk, knots=kts,integrator=integrator, message=verbose)
	# 	except ValueError:
	# 		try:
	# 			solver.solve(guess/10, tol=1e-9, number_knots=nk, knots=kts,integrator=integrator, message=verbose)
	# 		except ValueError:
	# 			pass
	# 	except AssertionError:
	# 		try:
	# 			solver.solve(guess/10, tol=1e-9, number_knots=nk, knots=kts,integrator=integrator, message=verbose)
	# 		except ValueError:
	# 			pass
	# 		except AssertionError:
	# 			try:
	# 				solver.solve(guess*10, tol=1e-9, number_knots=nk, knots=kts,integrator=integrator, message=verbose)
	# 			except ValueError:
	# 				pass
	# 			except AssertionError:
	# 				pass

	# except ValueError as e2:
	# 	print(e2)
	# 	try:
	# 		solver.solve(guess*10, tol=1e-9, number_knots=nk, knots=kts,integrator=integrator, message=verbose)
	# 	except AssertionError:
	# 		try:
	# 			solver.solve(guess/10, tol=1e-9, number_knots=nk, knots=kts,integrator=integrator, message=verbose)
	# 		except ValueError:
	# 			pass
	# 	except ValueError:
	# 		pass
 
	# Store
	sol1 = solver.solution
	# try:
	# 	sol1 = solver.solution
	# except AttributeError:
	# 	return {'solved': False}
	xs = sol1.index.values
	ys = sol1['$\\mu(x)$'].values
	thetas = sol1['$\\theta(x)$'].values
	ws = sol1['$w(x)$'].values
	pis = sol1['$\\pi(x)$'].values

	# For capital lambdify k_star
	eval_k = sym.lambdify((x,y,l,r), k_sol, modules='numpy')
	ks = eval_k(xs,ys,thetas,1.0)

	def discrete_pdf(x0,x1):
		p_cdf = productivity_cdf.subs({'loc2': 0.0, 'mu2': -1.8316, 'sigma2': 4.6553})
		y_cdf = sym.lambdify(x,p_cdf,modules = [{'ImmutableMatrix': np.array, 'erf': erf}, 'numpy'])
		return y_cdf(x1)-y_cdf(x0)
	
	xdist = (xs[1:]-xs[:-1])/2
	xdist = np.hstack((xdist[0],xdist))
	xsd = discrete_pdf(xs-xdist,xs+xdist)
	pdf_dis = xsd/np.sum(xsd)

	return {"solved":len(xs)==6000,"xs":xs , "ys":ys , "thetas":thetas , "ws": ws, "pis": pis, "ks": ks, "solution": sol1, "pdf_dis":pdf_dis, "F":F, "PAM": assort}

# PART II: Auxiliary functions # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

def derivatives(solution, country):
	"""
	Calcualtes the value of the cross-derivates of the production function at the solution points given.

	Convient for checking that the PAM condition holds: Fxy Flr > Fly Fxr (plot_PAM() below)

	Inputs
	------
	solution (dic)	: a dictionary from solve() above.
	country (str)	: "R" or "P" for the US or poor country baseline

	Outputs
	-------
	A dictionary with the derivatives:
		- Fxy
		- Flr
		- Fyl
		- Fxr
	"""
	# Unpacking
	F = solution["F"]
	xs,ys,thetas = solution["xs"],solution["ys"],solution["thetas"]
	R, l, r, rho, gamma, eta, L, A, kappa, p = sym.var('R, l, r, rho, gamma, eta, L, A, kappa, p')
	# Derivatives
	Fxy = sym.lambdify((x,y,l,r,A,kappa,eta,gamma,rho,R,p),F.diff(x,y), modules='numpy')
	Flr = sym.lambdify((x,y,l,r,A,kappa,eta,gamma,rho,R,p),F.diff(l,r), modules='numpy')
	Fyl = sym.lambdify((x,y,l,r,A,kappa,eta,gamma,rho,R,p),F.diff(y,l), modules='numpy')
	Fxr = sym.lambdify((x,y,l,r,A,kappa,eta,gamma,rho,R,p),F.diff(x,r), modules='numpy')

	if country=="P":
		R,A,p = 0.3958,0.3987,0.5209
	elif country=="R":
		R,A,p = 0.13099,1.0,0.3159
	dxy = Fxy(xs,ys,thetas,1.0,A,1.0,0.89,0.5,0.25,R,p)
	dlr = Flr(xs,ys,thetas,1.0,A,1.0,0.89,0.5,0.25,R,p)
	dyl = Fyl(xs,ys,thetas,1.0,A,1.0,0.89,0.5,0.25,R,p)
	dxr = Fxr(xs,ys,thetas,1.0,A,1.0,0.89,0.5,0.25,R,p)
	
	return {"Fxy": dxy, "Flr": dlr, "Fyl": dyl, "Fxr":dxr}

def get_kl_ratio(land,capital,bins,pdf_dis):
	"""
	Calculates the capital to land ratio distribution, given a set of bins, normalized by the highest bin.

	Convinient for replicating the histogram in A&R.

	Inputs
	-----
	land (np.array)		: array of equilibrium land choices (form solve())
	capital (np.array)	: array of equilibrium capital choices (form solve())
	bins (list or array): increasing array with bin edges
	pdf_dis (np.array)	: array with the weights of each point (form solve()), same length as land and capital

	Output
	------
	np.array with the values of the average k/l ratio for each bin, divided by that of the last bin.
	"""
	total_land = np.sum(land*pdf_dis)
	total_land_m = np.sum(farm_grid*y_prob)

	land_bin = np.zeros(13)
	k_bin = np.zeros(13)
	land_bin_m = np.zeros(13)
	k_bin_m = np.zeros(13)
	kl_bar = np.zeros(13)
	kl_bar_model = np.zeros(13)

	linx = 0
	b = 0
	for i in range(len(ks)):    
		if land[i]>=bins[b]:
			land_bin[b] = np.sum(land[linx:i]*pdf_dis[linx:i])/total_land
			k_bin[b] = np.sum(ks[linx:i]*pdf_dis[linx:i])
			kl_bar[b] = k_bin[b]/land_bin[b]
			linx = i
			b +=1
	# Add last bin
	land_bin[b] = np.sum(land[linx:i]*y_prob[linx:i])/total_land
	k_bin[b] = np.sum(ks[linx:i]*y_prob[linx:i])
	kl_bar[b] = k_bin[b]/land_bin[b]

	kl_bar = (kl_bar/kl_bar[-1])[1:]

	return kl_bar

# PART III: Plotting modules # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

def plot_histogram(sol_R1,sol_P1,dataR,dataP,save_in=None):
	"""
	Produces a histogram comparing the results of the model with A&R (2014).
	It does so for the US calibration and the Poor country (Aggregate Factors) calibration.

	Inputs
	------
	sol_R1 (dic)	: a dictionary solution from solve(), rich country ("R") parameters.
	sol_P1 (dic)	: a dictionary solution from solve(), rich country ("R") parameters.
	dataR (np.array): array with the results from the original model: (y,pdf(y),l,k,Y) from calibration.m
	dataP (np.array): array with the results from the original model: (y,pdf(y),l,k,Y) from AGGFAC.m
	save_in (str)	: (optional) name of the file to save the plot.

	Output
	------
	A matplotlib plot.

	"""
	y_grid, y_prob, farm_grid, k_grid, output_m = dataR
	y_grid_p, y_prob_p, farm_grid_p, k_grid_p, output_m_p = dataP
	xs,thetas, ks, pdf = sol_R1['xs'],1/sol_R1['thetas'],sol_R1['ks'],sol_R1['pdf_dis']
	xs2,thetas2, ks2, pdf2 = sol_P1['xs'],1/sol_P1['thetas'],sol_P1['ks'],sol_P1['pdf_dis']

	bins = (0,4,20,28,40,57,73,89,105,202,405,809,4000)
	uniform = np.ones(6000)*1/6000
	pdf_dis = sol_R1['pdf_dis']
	plt.figure(figsize=(15,6))

	plt.subplot(1,2,1)
	plt.title("US Baseline", fontsize=16)
	h,b = np.histogram(thetas, bins, weights=pdf)
	hm,bm = np.histogram(farm_grid, bins, weights=y_prob)
	plt.bar(np.arange(len(h)),h,width=0.4,color=sn.xkcd_rgb["windows blue"],label="Matching")
	plt.bar(np.arange(len(hm))+0.5,hm,width=0.4,color=sn.xkcd_rgb["pale yellow"],label="A&R")

	plt.xticks(np.linspace(0.5,12.5,13), ('<4','4-20','20-28','28-40','40-57','57-73','73-89','89-105',
			   '105-202','202-405','405-809','+809'),rotation='vertical', fontsize=14)
	plt.yticks(fontsize=14)
	plt.xlim(0,12)
	plt.ylim(0,0.35)
	plt.legend(fontsize=16,frameon=True)

	plt.subplot(1,2,2)
	plt.title("Poor country (less TFP and Land)", fontsize=16)
	h2,b2 = np.histogram(thetas2, bins, weights=pdf2)
	hm2,bm2 = np.histogram(farm_grid_p, bins, weights=y_prob_p)
	plt.bar(np.arange(len(h2)),h2,width=0.4,color=sn.xkcd_rgb["windows blue"],label="Matching")
	plt.bar(np.arange(len(hm2))+0.5,hm2,width=0.4,color=sn.xkcd_rgb["pale yellow"],label="A&R")
	plt.xticks(np.linspace(0.5,12.5,13), ('<4','4-20','20-28','28-40','40-57','57-73','73-89','89-105',
			   '105-202','202-405','405-809','+809'),rotation='vertical', fontsize=14)
	plt.yticks(fontsize=16)
	plt.legend(fontsize=16,frameon=True)
	plt.xlim(0,12)

	plt.tight_layout()

	if save_in!=None:
		plt.savefig(save_in, dpi=None, facecolor='w', edgecolor='w',
				orientation='portrait', papertype=None, format='pdf',
				transparent=False, bbox_inches="tight", pad_inches=0.1,
				frameon=None)
	plt.show()


def plot_comparative_land_choice(solsR,labelsR,solsP,labelsP,color=True,logs=True,save_in=None):
	"""
	Generates Report4.pdf and Report4b.pdf

	If soldP=None and logs=False, it delivers two plots: one with the normal scaling and one the log scaling.
	(Report4b.pdf)
	"""
	if color:
		sn.set_palette("colorblind",len(solsR)+2)
		alplist = np.ones(len(solsR))
	else:
		sn.set_palette("Greys_r",len(solsR)+2)
		alplist = np.ones(len(solsR))
	
	plt.figure(figsize=(12,6))
	allsols = zip(solsR,labelsR,alplist)
	plt.subplot(121)
	maxs = []
	mins = []
	for sol in allsols:
		plt.plot(sol[0]['xs'],1/sol[0]['thetas'],label=sol[1], alpha=sol[2])
		maxs.append(max(1/sol[0]['thetas']))
		mins.append(min(1/sol[0]['thetas'][sol[0]['xs']>=0.1]))
	plt.xlabel("farmer skill ($y$)", fontsize=16)
	plt.ylabel("farm size", fontsize=16)
	if logs:
		plt.xscale('log')
		plt.yscale('log')
		plt.xlim(0.1,100)
		plt.ylim(min(mins), max(maxs))
		tickies = np.logspace(np.log10(min(mins)), np.log10(max(maxs)),10)
		plt.yticks(tickies,np.round_(tickies, decimals=1),fontsize=12)
	else:
		plt.yticks(fontsize=12)
		plt.xlim(-0.5,100)
		plt.ylim(0,2500)
	# Same country plots
	if solsP==None:
		plt.subplot(122)
		maxs = []
		mins = []
		for sol in allsols:
			plt.plot(sol[0]['xs'],1/sol[0]['thetas'],label=sol[1], alpha=sol[2])
			maxs.append(max(1/sol[0]['thetas']))
			mins.append(min(1/sol[0]['thetas'][sol[0]['xs']>=0.1]))
		plt.xlabel("farmer skill ($y$)", fontsize=16)
		plt.ylabel("farm size", fontsize=16)
		plt.xscale('log')
		plt.yscale('log')
		plt.xlim(0.1,100)
		plt.ylim(min(mins), max(maxs))
		tickies = np.logspace(np.log10(min(mins)), np.log10(max(maxs)),10)
		plt.yticks(tickies,np.round_(tickies, decimals=1),fontsize=12)

	else: 
		allsols = zip(solsP,labelsP,alplist)
		plt.subplot(122)
		maxs = []
		mins = []
		for sol in allsols:
			plt.plot(sol[0]['xs'],1/sol[0]['thetas'],label=sol[1], alpha=sol[2])
			maxs.append(max(1/sol[0]['thetas']))
			mins.append(min(1/sol[0]['thetas'][sol[0]['xs']>=0.1]))
		if logs:
			plt.xscale('log')
			plt.yscale('log')
			plt.xlim(0.1,100)
			plt.ylim(min(mins), max(maxs))
			tickies = np.logspace(np.log10(min(mins)), np.log10(max(maxs)),10)
			plt.yticks(tickies,np.round_(tickies, decimals=1),fontsize=12)
		else:
			plt.yticks(fontsize=12)
			plt.xlim(0,100)
			plt.ylim(0,300)
	plt.xlabel("farmer skill ($y$)", fontsize=16)
	plt.ylabel("farm size", fontsize=16)
	plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., frameon=True, fontsize=14)

	plt.tight_layout()
	if save_in != None:
		plt.savefig(save_in, dpi=None, facecolor='w', edgecolor='w',
				orientation='portrait', papertype=None, format='pdf',
				transparent=False, bbox_inches="tight", pad_inches=0.1,
				frameon=None)

	plt.show()

def plot_land_choice(sols,labels,color=True,logs=True,yscale=None,save_in=None):
	"""
	Generates Report4.pdf and Report4b.pdf

	If soldP=None and logs=False, it delivers two plots: one with the normal scaling and one the log scaling.
	(Report4b.pdf)
	"""
	if color:
		sn.set_palette("colorblind",len(sols)+2)
		alplist = np.ones(len(sols))
	else:
		sn.set_palette("Greys_r",len(sols)+2)
		alplist = np.ones(len(sols))
	
	plt.figure(figsize=(8,6))
	allsols = zip(sols,labels,alplist)
	maxs = []
	mins = []
	for sol in allsols:
		plt.plot(sol[0]['xs'],1/sol[0]['thetas'],label=sol[1], alpha=sol[2])
		maxs.append(max(1/sol[0]['thetas']))
		mins.append(min(1/sol[0]['thetas'][sol[0]['xs']>=0.1]))
	plt.xlabel("farmer skill ($y$)", fontsize=16)
	plt.ylabel("farm size", fontsize=16)
	if logs:
		plt.xscale('log')
		plt.yscale('log')
		plt.xlim(0.1,100)
		plt.ylim(min(mins), max(maxs))
		tickies = np.logspace(np.log10(min(mins)), np.log10(max(maxs)),10)
		plt.yticks(tickies,np.round_(tickies, decimals=1),fontsize=12)
	else:
		plt.yticks(fontsize=12)
		plt.xlim(0.1,100)
		if yscale!=None:
			plt.ylim(0,yscale)
		else:
			plt.ylim(0,max(maxs))
	plt.xlabel("farmer skill ($y$)", fontsize=16)
	plt.ylabel("farm size", fontsize=16)
	legend=plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., frameon=True, fontsize=14, title='Land quality spread')
	plt.setp(legend.get_title(),fontsize=16)
	
	plt.tight_layout()
	if save_in != None:
		plt.savefig(save_in, dpi=None, facecolor='w', edgecolor='w',
				orientation='portrait', papertype=None, format='eps',
				transparent=False, bbox_inches="tight", pad_inches=0.1)

	plt.show()


# Other plotting modules (director's cut) # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# These plots are not in the paper but may be used in the notebook # # # # # # # # # # # # # # # # # # # # # # # 

def plot_kl_ratio(solsR,labelsR,solsP,labelsP,color=True,save_in=None, showw=True):
	"""
	Generates Report5.pdf
	"""
	if color:
		sn.set_palette("colorblind",len(solsR))
		alplist = np.ones(len(solsR))
	else:
		sn.set_palette("Greys_r",len(solsR)+2)
		alplist = np.ones(len(solsR))

	allsols = zip(solsR,labelsR,alplist)

	plt.figure(figsize=(12,6))
	plt.subplot(121)
	for sol in allsols:
		ratio = (sol[0]['ks']/(1/sol[0]['thetas']))
		plt.plot(sol[0]['xs'],ratio,label=sol[1], alpha=sol[2])
	plt.xlabel("Farmer Skill", fontsize=16)
	plt.ylabel('$k/l$', fontsize=16)
	plt.xticks(fontsize=14)
	plt.yticks(fontsize=14)
	plt.ylabel('$k/l$ normalized', fontsize=16)
	plt.xscale('log')
	plt.ylim(0,1)

	allsols = zip(solsP,labelsP,alplist)
	plt.subplot(122)
	for sol in allsols:
		ratio = (sol[0]['ks']/(1/sol[0]['thetas']))
		plt.plot(sol[0]['xs'],ratio,label=sol[1], alpha=sol[2])
	plt.xlabel("Farmer Skill", fontsize=16)
	plt.ylabel('$k/l$', fontsize=16)
	plt.xticks(fontsize=14)
	plt.yticks(fontsize=14)
	plt.ylim(0,1)
	plt.ylabel('$k/l$ normalized', fontsize=16)
	plt.xscale('log')
	plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., frameon=True, fontsize=14)

	if save_in != None:
		plt.savefig(save_in, dpi=None, facecolor='w', edgecolor='w',
				orientation='portrait', papertype=None, format='pdf',
				transparent=False, bbox_inches="tight", pad_inches=0.1,
				frameon=None)
	if showw:
		plt.show()

def plot_dischange(sol_R1,sol_Rx,xrange=(0.9,1.1),extras=None,save_in=None,allplots=True,color=True,uc=cr, show=True):

	"""
	Plots the change in the distriution of firm size from an original distribution (sol_R1) to one with increased spread of x (sol_Rx).
	"""
	xs1,thetas1, ks1, pdf1 = sol_R1['xs'],1/sol_R1['thetas'],sol_R1['ks'],sol_R1['pdf_dis']
	xs10,thetas10, ks10, pdf10 = sol_Rx['xs'],1/sol_Rx['thetas'],sol_Rx['ks'],sol_Rx['pdf_dis']
	if extras != None:
		xs20,thetas20, ks20, pdf20 = extras[0]['xs'],1/extras[0]['thetas'],extras[0]['ks'],extras[0]['pdf_dis']

	if color==True:
		sty = ['-','-','-']
		c = uc
	else:
		sty = ['-','-','-']
		c = 'black'

	cdf10 = np.cumsum(pdf10)
	cdf1 = np.cumsum(pdf1)
	if extras != None:
		cdf20 = np.cumsum(pdf20)
	
	plt.figure(figsize=(12,6))
	
	plt.subplot(1,2,1)
	plt.plot(thetas1,pdf1,color=c, ls=sty[0],alpha=0.35)
	if extras != None:
		plt.plot(thetas20,pdf20,color=c, ls=sty[1],alpha=0.7)
	plt.plot(thetas10,pdf10,ls=sty[2],color=c)
	plt.ylabel('pdf', fontsize=18)
	plt.xlabel('farm size', fontsize=18)
	plt.xticks(fontsize=14)
	plt.yticks(fontsize=14)
	plt.xscale('log')

	plt.subplot(1,2,2)
	plt.plot(thetas1,cdf1,label='$x=1$',c=c,ls=sty[0],alpha=0.35)
	if extras != None:
		plt.plot(thetas20,cdf20,label='$x='+str(extras[1])+'$',c=c,ls=sty[1],alpha=0.7)
	plt.plot(thetas10,cdf10,label='$x='+str(xrange)+'$',ls=sty[2],c=c)
	plt.ylabel('cdf', fontsize=18)
	plt.xlabel('farm size', fontsize=18)
	plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., frameon=True,fontsize=14)
	plt.xscale('log')
	plt.xticks(fontsize=14)
	plt.yticks(fontsize=14)
	plt.ylim(0,1)

	if save_in != None:
		plt.savefig(save_in, dpi=None, facecolor='w', edgecolor='w', orientation='portrait', papertype=None, 
			format='pdf', transparent=False, bbox_inches='tight', pad_inches=0.1, frameon=None)
	if show:
		plt.show()

def plot_k_fit(Grid,funcs,params,best_fit):
	"""
	Plots the fit of the approxiamted solution to k(x,y,l,r).

	Module used by get_k().

	"""
	# Unpacking
	Kgrid, xgrid, ygrid, lgrid = Grid
	params_ten3o,params_ten3,params_s = params
	res_ten3o, res_ten3, pol_shorcut = funcs
	cls = ['black','black','black']
	cls[best_fit] = 'green'

	#Plotting
	plt.figure(figsize=(12,3))
	plt.subplot(1,3,1)
	plt.plot(Kgrid.flatten(), color="grey", alpha=0.8)
	plt.plot(res_ten3o(params_ten3o,np.array([0.,]),xgrid,ygrid,lgrid), alpha=0.5)
	plt.title("Tensors 2nd",color=cls[0])
	plt.xlim(0,len(Kgrid.flatten()))

	plt.subplot(1,3,2)
	plt.plot(Kgrid.flatten(), color="grey", alpha=0.8)
	plt.plot(res_ten3(params_ten3,np.array([0.,]),xgrid,ygrid,lgrid), alpha=0.5)
	plt.title("Tensors 3rd",color=cls[1])
	plt.xlim(0,len(Kgrid.flatten()))

	plt.subplot(1,3,3)
	plt.plot(Kgrid.flatten(), color="grey", alpha=0.8)
	plt.plot(pol_shorcut(params_s,np.array([0.,]),xgrid.flatten(),ygrid.flatten(),lgrid.flatten()), alpha=0.5)
	plt.title("Polynomial approx",color=cls[2])
	plt.xlim(0,len(Kgrid.flatten()))

	plt.show()


def plot_PAM(derivatives, xs_v, zoom=None, save=False):
	"""
	Plots the PAM condition in equilibrium: Fxy Flr - Fly Fxr > 0 (for PAM) or  < 0 (for NAM)

	"""
	# Unpack
	if len(xs_v)==2:
		dF, dF2 = derivatives
		xs, xs2 = xs_v
		dxy2, dlr2, dyl2,dxr2 = dF2["Fxy"], dF2["Flr"], dF2["Fyl"], dF2["Fxr"]
		plt.subplot(1,2,1)
	else:
		dF = derivatives
		xs = xs_v

	dxy, dlr, dyl,dxr = dF["Fxy"], dF["Flr"], dF["Fyl"], dF["Fxr"]
		
	if len(xs_v)==2:
		plt.subplot(1,2,1)
	PAM_check = dxy*dlr-dyl*dxr
	plt.plot(xs,PAM_check)
	plt.xscale("log")
	if zoom!= None:
		plt.ylim(zoom[0],zoom[1])
	#plt.axvline(cut, color="grey", ls="--")
	plt.xlabel("$\\tilde{x}$", fontsize=20)
	plt.ylabel("PAM condition at equilibrium", fontsize=14)
	plt.title("Checking PAM")

	if len(xs_v)==2:
		plt.subplot(1,2,1)
		plt.subplot(1,2,2)
		PAM_check2 = dxy2*dlr2-dyl2*dxr2
		plt.plot(xs2,PAM_check2)
		plt.xscale("log")
		if zoom!= None:
			plt.ylim(zoom[0],zoom[1])
		#plt.ylim(-1,1)
		#plt.axvline(cut, color="grey", ls="--")
		plt.xlabel("$\\tilde{x}$", fontsize=20)
		plt.title("Less A")
	if save:
		plt.savefig("PAM.png")
	plt.show()
