import re # for 'is_file'
import os

from random import randint
from random import random as _random
def rand(n=1):return _random()*n
def random(n=1):return _random()*n
def random_array(l):return np.random.rand(l) # (0,1) x,y don't work ->
def random_matrix(x,y):return np.random.rand(x,y) # (0,1) !

def pick(xs):
    return xs[randint(len(xs))]

def readlines(source):
    print("open(source).readlines()")
    return map(str.strip,open(source).readlines())

def reverse(x):
    y=x.reverse()
    return y or x

def h(x):
  help(x)
  
def log(msg):
  print(msg)

def fold(self,x,fun):
  if not callable(fun): 
    fun,x=x,fun
  return reduce(fun, self, x)

def last(xs):
  return xs[-1]
  
def Pow(x,y):
    return x**y

def is_string(s):
    # import extensions
    class xstr(str):pass
    return isinstance(s,str) or isinstance(s,xstr) or isinstance(s,unicode) #or issubclass(s,str) or issubclass(s,unicode)

def flatten(l):
    if isinstance(l, list) or isinstance(l, tuple):
        for k in l:
            if isinstance(k, list):
                l.remove(k)
                l.append(*k)
    else:return [l]
    # verbose("NOT flattenable: %s"%s)
    return l


def square(x):
    if isinstance(x,list): return map(square,x)
    return x*x

def puts(x):
    print(x)
    return x

def increase(x):
    import nodes
    # if isinstance(x, dict)
    if isinstance(x, nodes.Variable):
        x.value=x.value+1
        return x.value
    return x+1

def grep(xs, x):
    # filter(lambda y: re.match(x,y),xs)
    if isinstance(x,list):
        return filter(lambda y: x[0] in str(y),xs)
    return filter(lambda y: x in str(y),xs)

def ls(mypath="."):
    from extensions import xlist
    return xlist(os.listdir(mypath))

def length(self):
    return len(self)

def say(x):
    print(x) 
    os.system("say '%s'"%x)

def bash(x):
    os.system(x)

def beep():
    print("\aBEEP ")

def beep(bug=True):
    print("\aBEEP ")
    import angle
    if not angle.testing:
        import os
        os.system("say 'beep'")
    return 'beeped'


def match_path(p):
    if(not isinstance(p,str)):return False
    m = re.search(r'^(\/[\w\'\.]+)',p)
    if not m: return False
    return m


def is_file(p, must_exist=True):
    if(not isinstance(p,str)):return False
    if re.search(r'^\d*\.\d+',p): return False
    if re.match(r'^\d*\.\d+',str(p)): return False
    m = re.search(r'^(\/[\w\'\.]+)',p)
    m = m or re.search(r'^([\w\/\.]*\.\w+)',p)
    if not m: return False
    return must_exist and m and os.path.isfile(m.string) or m


def is_dir(x, must_exist=True):
    #(the.string+" ").match(r'^(\')?([^\/\\0]+(\')?)+ ')
    m = match_path(x)
    return must_exist and m and os.path.isdirectory(m[0]) or m

# def print(x # debug!):
# print x
#   print "\n"
#   x

def grep(xs,pattern):
    return filter(pattern, xs)

print("extension functions loaded")
