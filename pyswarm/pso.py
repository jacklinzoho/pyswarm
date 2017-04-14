from functools import partial
import numpy as np
#using this instead on windows.

from time import sleep

def async_particle((id, obj, lb, ub, is_feasible, omega, phip, phig, g, minstep)):
    #windows weirdness.
    #asynchronous particle.  this runs until the main thread tells it to stop.
    #start by finding a valid x within confines 
    #and initializing all the vars
    print("start "+str(id))
    D = len(lb)
    x = np.random.rand(D)
    x = lb + x*(ub - lb)

    while not is_feasible(x):
        x = np.random.rand(D)  # particle positions
        x = lb + x*(ub - lb)

    fx = obj(x)
    p = list(x)  # best particle positions
    fp = fx  # current particle function values
    #append it to the global list
    g.add(x, fx, id)
    vhigh = np.abs(ub - lb)
    vlow = -vhigh
    v = vlow + np.random.rand(D)*(vhigh - vlow)  # particle initial velocity

    #todo: add termination condition 
    #what do i do if particle stuck in local minima?
    while not g.end:
        sleep(0.1) #this is for testing purposes.  
        rp = np.random.uniform(size=D)
        rg = np.random.uniform(size=D)

        # Update the particles velocities
        v = omega*v + phip*rp*(p - x) + phig*rg*(g.g - x)
        # maintain minimum velocity inside the particle
        if np.linalg.norm(v) < minstep:
            return 0
        
        # Update the particles' positions
        x = x + v
        # Correct for bound violations
        maskl = x < lb
        masku = x > ub
        x = x*(~np.logical_or(maskl, masku)) + lb*maskl + ub*masku

        # Store particle's best position (if constraints are satisfied)
        if is_feasible(x):
            fx = obj(x)
            if fx < fp:
                p = list(x)
                fp = fx
            g.add(x, fx, id) # we add all results to the global list, not just particle best.
                               # makes it easier to do post-processing
        print(x)
        print(fx)
    return 0

class async_g():
    #store:
    #a counter for how many evals have been completed; a list of positions; a list of results; best; position of best.
    #thread safe.
    #maybe I should add a way to pre-load a list?
    def __init__(self,D):
        from multiprocessing.dummy import Lock
        self.xlog = {}
        self.fxlog = {}
        for i in range(D):
            self.xlog[str(i)] = []
            self.fxlog[str(i)] = []
        #print self.xlog
        #print self.fxlog
        self.g = np.random.rand(D) #position of global best
        self.fg = np.inf #cost of global best
        self.lock = Lock()
        self.count = 0
        self.end = False
        
    def add(self, x, fx, id):
        with self.lock:
            self.xlog[str(id)].append(list(x))
            self.fxlog[str(id)].append(fx)
            self.count = self.count +1
            if fx < self.fg:
                self.g = list(x)
                self.fg = fx



def cleanup(processes_list):
    print("running pre-exit cleanup...")
    for p in processes_list: # list of your processes
        p.join()
        p.terminate()
    print("done.")

def _obj_wrapper(func, args, kwargs, x):
    return func(x, *args, **kwargs)

def _is_feasible_wrapper(func, x):
    return np.all(func(x)>=0)

def _cons_none_wrapper(x):
    return np.array([0])

def _cons_ieqcons_wrapper(ieqcons, args, kwargs, x):
    return np.array([y(x, *args, **kwargs) for y in ieqcons])

def _cons_f_ieqcons_wrapper(f_ieqcons, args, kwargs, x):
    return np.array(f_ieqcons(x, *args, **kwargs))
    
def pso(func, lb, ub, ieqcons=[], f_ieqcons=None, args=(), kwargs={}, 
        swarmsize=100, omega=0.5, phip=0.5, phig=0.5, maxiter=100, 
        minstep=1e-8, minfunc=1e-8, debug=False, processes=1,
        particle_output=False, async=True):
    """
    Perform an asynchronous particle swarm optimization (PSO)
    Set async=False to mimic behavior of pyswarm 0.7 (synchronous)
   
    Parameters
    ==========
    func : function
        The function to be minimized
    lb : array
        The lower bounds of the design variable(s)
    ub : array
        The upper bounds of the design variable(s)
   
    Optional
    ========
    ieqcons : list
        A list of functions of length n such that ieqcons[j](x,*args) >= 0.0 in 
        a successfully optimized problem (Default: [])
    f_ieqcons : function
        Returns a 1-D array in which each element must be greater or equal 
        to 0.0 in a successfully optimized problem. If f_ieqcons is specified, 
        ieqcons is ignored (Default: None)
    args : tuple
        Additional arguments passed to objective and constraint functions
        (Default: empty tuple)
    kwargs : dict
        Additional keyword arguments passed to objective and constraint 
        functions (Default: empty dict)
    swarmsize : int
        The number of particles in the swarm (Default: 100)
    omega : scalar
        Particle velocity scaling factor (Default: 0.5)
    phip : scalar
        Scaling factor to search away from the particle's best known position
        (Default: 0.5)
    phig : scalar
        Scaling factor to search away from the swarm's best known position
        (Default: 0.5)
    maxiter : int
        The maximum number of iterations for the swarm to search (Default: 100)
    minstep : scalar
        The minimum stepsize of swarm's best position before the search
        terminates (Default: 1e-8)
    minfunc : scalar
        The minimum change of swarm's best objective value before the search
        terminates (Default: 1e-8)
    debug : boolean
        If True, progress statements will be displayed every iteration
        (Default: False)
    processes : int
        The number of processes to use to evaluate objective function and 
        constraints (default: 1)
    particle_output : boolean
        Whether to include the best per-particle position and the objective
        values at those.
   
    Returns
    =======
    g : array
        The swarm's best known position (optimal design)
    f : scalar
        The objective value at ``g``
    p : array
        The best known position per particle
    pf: arrray
        The objective values at each position in p
   
    """
   
    assert len(lb)==len(ub), 'Lower- and upper-bounds must be the same length'
    assert hasattr(func, '__call__'), 'Invalid function handle'
    lb = np.array(lb)
    ub = np.array(ub)
    assert np.all(ub>lb), 'All upper-bound values must be greater than lower-bound values'
   
    vhigh = np.abs(ub - lb)
    vlow = -vhigh

    # Initialize objective function
    obj = partial(_obj_wrapper, func, args, kwargs)
    
    # Check for constraint function(s) #########################################
    if f_ieqcons is None:
        if not len(ieqcons):
            if debug:
                print('No constraints given.')
            cons = _cons_none_wrapper
        else:
            if debug:
                print('Converting ieqcons to a single constraint function')
            cons = partial(_cons_ieqcons_wrapper, ieqcons, args, kwargs)
    else:
        if debug:
            print('Single constraint function given in f_ieqcons')
        cons = partial(_cons_f_ieqcons_wrapper, f_ieqcons, args, kwargs)
    is_feasible = partial(_is_feasible_wrapper, cons)

    # Initialize the multiprocessing module if necessary
    if processes > 1:
        import multiprocessing.dummy
        pool = multiprocessing.dummy.Pool(processes=processes)
            
    # Initialize the particle swarm ############################################
    S = swarmsize
    D = len(lb)  # the number of dimensions each particle has
    x = np.random.rand(S, D)  # particle positions
    v = np.zeros_like(x)  # particle velocities
    p = np.zeros_like(x)  # best particle positions
    fx = np.zeros(S)  # current particle function values
    fs = np.zeros(S, dtype=bool)  # feasibility of each particle
    fp = np.ones(S)*np.inf  # best particle function values
    g = []  # best swarm position
    fg = np.inf  # best swarm position starting value
    
    # Initialize the particle's position
    x = lb + x*(ub - lb)
    
    #----- async code starts here ------        
    # async assumes multiple processes.  it will work even with single, but we'll still use the same implementation.
    # also, swarmsize is not used.  instead, swarmsize is equal to processes.  I'll fix this later.
    if async:
        # best swarm position

        g = async_g(len(lb))
        last_count = g.count
        last_fg = g.fg
        last_g = list(g.g)
        
        args = []
        for i in range(processes):
            args.append((i, obj, lb, ub, is_feasible, omega, phip, phig, g, minstep))
        pool.map(async_particle, args)
        while True:
            #main loop
            sleep(0.05)
            #watch the list for the following conditions every 1 second:
            new_count = g.count
            if g.fg < last_fg:
                if debug:
                    print('New best for swarm at iteration {:}: {:} {:}'\
                            .format(int(new_count/processes + 1), g.g, g.fg))

                if g.fg - last_fg < minfunc:
                    print('Stopping search: Swarm best objective change less than {:}'\
                        .format(minfunc))
                    g.end = True
                    sleep(1)
                    pool.close()
                    pool.join()
                    if particle_output:
                        return g.g, g.fg
                        #return p_min, fp[i_min], p, fp
                    else:
                        return g.g, g.fg             
                
                # the async version probably needs to be tighter with the minstep than pyswarm
                # since g is updated more often and the probability of getting an update within minstep is higher
                stepsize = np.linalg.norm(new_g - last_g)
                if stepsize <= minstep:
                    print('Stopping search: Swarm best position change less than {:}'\
                        .format(minstep))
                    g.end = True
                    sleep(1)
                    pool.close()
                    pool.join()
                    if particle_output:
                        return g.g, g.fg
                        #return p_min, fp[i_min], p, fp
                    else:
                        return g.g, g.fg             

                #number of evals performed so far / swarm size + 1 ~= number of iterations in synchronized pso
                if int(new_count/processes+1) >=maxiter:
                    print('Stopping search: maximum iterations reached --> {:}'.format(maxiter))
                    g.end = True
                    sleep(1)
                    pool.close()
                    pool.join()
                    if particle_output:
                        return g.g, g.fg
                        #return p_min, fp[i_min], p, fp
                    else:
                        return g.g, g.fg             
                last_fg = g.fg
                last_g = list(g.g)
        
        # poll the global list
        # compare current with last
        # we don't really have iterations as such, so 
        # here we adapt the convention that iteration = int(len(g.pl)/swarmsize+1)
        # by this definition probably it needs more 'iterations' than pyswarm
        # but walltime is what's really important, and it wins by that metric.
        # particle_output is not supported and ignored.
        
    #---- everything below this line should be unchanged from pyswarm 0.7 ----

    if processes > 1:
        fx = np.array(mp_pool.map(obj, x))
        fs = np.array(mp_pool.map(is_feasible, x))
    else:
        for i in range(S):
            fx[i] = obj(x[i, :])
            fs[i] = is_feasible(x[i, :])
       
    # Store particle's best position (if constraints are satisfied)
    i_update = np.logical_and((fx < fp), fs)
    p[i_update, :] = x[i_update, :].copy()
    fp[i_update] = fx[i_update]

    # Update swarm's best position
    i_min = np.argmin(fp)
    if fp[i_min] < fg:
        fg = fp[i_min]
        g = p[i_min, :].copy()
    else:
        # At the start, there may not be any feasible starting point, so just
        # give it a temporary "best" point since it's likely to change
        g = x[0, :].copy()
       
    # Initialize the particle's velocity
    v = vlow + np.random.rand(S, D)*(vhigh - vlow)
       
    # Iterate until termination criterion met ##################################
    it = 1
    while it <= maxiter:
        rp = np.random.uniform(size=(S, D))
        rg = np.random.uniform(size=(S, D))

        # Update the particles velocities
        v = omega*v + phip*rp*(p - x) + phig*rg*(g - x)
        # Update the particles' positions
        x = x + v
        # Correct for bound violations
        maskl = x < lb
        masku = x > ub
        x = x*(~np.logical_or(maskl, masku)) + lb*maskl + ub*masku

        # Update objectives and constraints
        if processes > 1:
            fx = np.array(mp_pool.map(obj, x))
            fs = np.array(mp_pool.map(is_feasible, x))
        else:
            for i in range(S):
                fx[i] = obj(x[i, :])
                fs[i] = is_feasible(x[i, :])

        # Store particle's best position (if constraints are satisfied)
        i_update = np.logical_and((fx < fp), fs)
        p[i_update, :] = x[i_update, :].copy()
        fp[i_update] = fx[i_update]

        # Compare swarm's best position with global best position
        i_min = np.argmin(fp)
        if fp[i_min] < fg:
            if debug:
                print('New best for swarm at iteration {:}: {:} {:}'\
                    .format(it, p[i_min, :], fp[i_min]))

            p_min = p[i_min, :].copy()
            stepsize = np.sqrt(np.sum((g - p_min)**2))

            if np.abs(fg - fp[i_min]) <= minfunc:
                print('Stopping search: Swarm best objective change less than {:}'\
                    .format(minfunc))
                if particle_output:
                    return p_min, fp[i_min], p, fp
                else:
                    return p_min, fp[i_min]
            elif stepsize <= minstep:
                print('Stopping search: Swarm best position change less than {:}'\
                    .format(minstep))
                if particle_output:
                    return p_min, fp[i_min], p, fp
                else:
                    return p_min, fp[i_min]
            else:
                g = p_min.copy()
                fg = fp[i_min]

        if debug:
            print('Best after iteration {:}: {:} {:}'.format(it, g, fg))
        it += 1

    print('Stopping search: maximum iterations reached --> {:}'.format(maxiter))
    
    if not is_feasible(g):
        print("However, the optimization couldn't find a feasible design. Sorry")
    if particle_output:
        return g, fg, p, fp
    else:
        return g, fg

