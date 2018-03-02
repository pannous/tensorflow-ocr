# encoding: utf-8
# nocoding: interpy "string interpolation #{like ruby}"
# encoding=utf8
import io
import math
import sys
import inspect
import re  # for 'is_file'
import os
# import __builtin__
import numpy as np
from random import randint
from random import random as _random
import shutil

# from extension_functions import * MERGED BACK!

py2 = sys.version < '3'
py3 = sys.version >= '3'

true = True
false = False

pi = math.pi
E = math.e


def Max(a, b):
	if a > b:
		return a
	return b


def Min(a, b):
	if a > b:
		return b
	return a


def rand(n=1): return _random() * n


def random(n=1): return _random() * n


def random_array(l): return np.random.rand(l)  # (0,1) x,y don't work ->


def random_matrix(x, y): return np.random.rand(x, y)  # (0,1) !


def pick(xs):
	return xs[randint(len(xs))]


def reverse(x):
	y = x.reverse()
	return y or x


def h(x):
	help(x)


def log(msg):
	print(msg)


def fold(self, x, fun):
	if not callable(fun):
		fun, x = x, fun
	return reduce(fun, self, x)


def last(xs):
	return xs[-1]


def Pow(x, y):
	return x**y


def is_string(s):
	return isinstance(s, str) or isinstance(s, xstr) or isinstance(s, unicode)


# or issubclass(s,str) or issubclass(s,unicode)


def flatten(l):
	if isinstance(l, list) or isinstance(l, tuple):
		for k in l:
			if isinstance(k, list):
				l.remove(k)
				l.append(*k)
	else:
		return [l]
	# verbose("NOT flattenable: %s"%s)
	return l


def square(x):
	if isinstance(x, list): return map(square, x)
	return x * x


def puts(x):
	print(x)
	return x


def increase(x):
	import nodes
	# if isinstance(x, dict)
	if isinstance(x, nodes.Variable):
		x.value = x.value + 1
		return x.value
	return x + 1


def grep(xs, x):
	# filter(lambda y: re.match(x,y),xs)
	# return filter(pattern, xs)
	if isinstance(x, list):
		return filter(lambda y: x[0] in str(y), xs)
	return filter(lambda y: x in str(y), xs)


def ls(mypath="."):
	from extensions import xlist
	return xlist(os.listdir(mypath))


def length(self):
	return len(self)


def say(x):
	print(x)
	os.system("say '%s'" % x)


def bash(x):
	os.system(x)


def beep():
	print("\aBEEP ")


def beep(bug=True):
	print("\aBEEP ")
	import context
	if not context.testing:
		import os
		os.system("say 'beep'")
	return 'beeped'


def match_path(p):
	if not isinstance(p, str): return False
	m = re.search(r'^(/[\w\'.]+)', p)
	if not m: return []
	return m


def regex_match(a, b):
	NONE = "None"
	match = regex_matches(a, b)
	if match:
		try:
			return a[match.start():match.end()].strip()
		except:
			return b[match.start():match.end()].strip()
	return NONE


# RegexType= _sre.SRE_Pattern#type(r'')
MatchObjectType = type(re.search('', ''))
def typeof(x):
	print("type(x)")
	return type(x)

def regex_matches(a, b):
	if isinstance(a, re._pattern_type):
		return a.search(b)  #
	if isinstance(b, re._pattern_type):
		return b.search(a)
	if is_string(a) and len(a) > 0:
		if a[0] == "/": return re.compile(a).search(b)
	if is_string(b) and len(b) > 0:
		if b[0] == "/": return re.compile(b).search(a)

	try:
		b = re.compile(b)
	except:
		print("FAILED: re.compile(%s)" % b)
		b = re.compile(str(b))
	print(a)
	print(b)
	return b.search(str(a))  # vs


# return b.match(a) # vs search
# return a.__matches__(b) # other cases
# return a.contains(b)


def is_file(p, must_exist=True):
	if not isinstance(p, str): return False
	if re.search(r'^\d*\.\d+', p): return False
	if re.match(r'^\d*\.\d+', str(p)): return False
	m = re.search(r'^(\/[\w\'\.]+)', p)
	m = m or re.search(r'^([\w\/\.]*\.\w+)', p)
	if not m: return False
	return must_exist and m and os.path.isfile(m.string) or m


def is_dir(x, must_exist=True):
	# (the.string+" ").match(r'^(\')?([^\/\\0]+(\')?)+ ')
	m = match_path(x)
	return must_exist and m and (py3 and os.path.isdir(m[0])) or (py2 and os.path.isdirectory(m[0]))


# def print(x # debug!):
# print x
#   print "\n"
#   x

def is_a(self, clazz):
	if self is clazz: return True
	try:
		ok = isinstance(self, clazz)
		if ok: return True
	except Exception as e:
		print(e)

	className = str(clazz).lower()
	if className == str(self).lower(): return True  # KINDA

	if self.is_(clazz): return True
	return False


# print("extension functions loaded")


# from fractions import Fraction
# x = Fraction(22, 7) 	# Ruby: 22 / 7r 22r / 7

if py3:
	class file(io.IOBase):
		pass  # WTF python3 !?!?!?!?!??


	class xrange:  # WTF python3 !?!?!?!?!??
		pass


	class xstr(str):
		pass  # later


	class unicode(xstr):  # , bytes):  # xchar[] TypeError: multiple bases have instance lay-out conflict
		# Python 3 renamed the unicode type to str, the old str type has been replaced by bytes.
		pass

	# else: https://stackoverflow.com/questions/22098099/reason-why-xrange-is-not-inheritable-in-python
	#   class range(xrange):
	#     pass

else:  # Python 2 needs:
	class bytes(str):
		pass


	class char(str):
		pass


# char = char

class byte(str):
	pass


# byte= byte
file = file  # nice trick: native py2 class or local py3 class
unicode = unicode
xrange = xrange

if py2:
	import cPickle as pickle
else:
	import dill as pickle


# try: # py2
#   import sys
#   reload(sys)
#   sys.setdefaultencoding('utf8')
# except:
#   pass


def type_name(x):
	return type(x).__name__


def xx(y):
	if type_name(y).startswith('x'):   return y
	# if isinstance(y, xstr):   return y
	# if isinstance(y, xlist):   return y
	if isinstance(y, xrange): return xlist(y)
	if isinstance(y, bool):  return y  # xbool(y)
	if isinstance(y, list):  return xlist(y)
	if isinstance(y, str):   return xstr(y)
	if isinstance(y, unicode):   return xstr(y)
	if isinstance(y, dict):  return xdict(y)
	if isinstance(y, float): return xfloat(y)
	if isinstance(y, int):   return xint(y)
	if isinstance(y, file):   return xfile(y)
	if isinstance(y, char):  return xchar(y)
	if isinstance(y, byte):  return xchar(y)
	if py3 and isinstance(y, range): return xlist(y)
	print("No extension for type %s" % type(y))
	return y


extensionMap = {}


def extension(clazz):
	try:
		for base in clazz.__bases__:
			extensionMap[base] = clazz
	except:
		pass
	return clazz


# class Extension:
#     def __init__(self, base,b=None,c=None):
#         self.base=base
#
#     def __call__(self,clazz):
#         self.base()
#         import angle
#         angle.extensionMap[self.base]=clazz
#         print(angle.extensionMap)



class Class:
	pass


# not def(self):
#   False


# class Method(__builtin__.function):
#     pass

# file() is no supported in python3
# use open() instead

@extension
class xfile(file):
	# import fileutils

	# str(def)(self):
	#   path
	path = ""

	def name(self):
		return self.path

	def filename(self):
		return self.path

	def mv(self, to):
		os.rename(self.path, to)

	def move(self, to):
		os.rename(self.path, to)

	def copy(self, to):
		shutil.copyfile(self.path, to)

	def cp(self, to):
		shutil.copyfile(self.path, to)

	def contain(self, x):
		return self.path.index(x)

	def contains(self, x):
		return self.path.index(x)

	@staticmethod
	def delete():
		raise Exception("SecurityError: cannot delete files")

	# FileUtils.remove_dir(to_path, True)

	# @classmethod
	# def open(cls):return open(cls)
	# @classmethod
	# def read(cls):return open(cls)
	# @classmethod
	# def ls(cls):
	#     return os.listdir(cls)

	@staticmethod
	def open(x): return open(x)

	@staticmethod
	def read(x): return open(x)

	@staticmethod
	def ls(mypath="."):
		return xlist(os.listdir(mypath))


@extension
class File(xfile):
	pass


@extension
class Directory(file):  #
	# str(def)(self):
	#   path

	@classmethod
	def cd(path):
		os.chdir(path)

	def files(self):
		os.listdir(str(self))  # ?

	@classmethod
	def ls(path="."):
		os.listdir(path)

	@classmethod
	def files(path):
		os.listdir(path)

	def contains(self, x):
		return self.files().has(x)

	# Dir.cd
	#  Dir.glob "*.JPG"
	#
	# import fileutils
	#
	# def remove_leaves(dir=".", matching= ".svn"):
	#   Dir.chdir(dir) do
	#     entries=Dir.entries(Dir.pwd).reject (lambda e: e=="." or e==":" )
	#     if entries.size == 1 and entries.first == matching:
	#       print "Removing #{Dir.pwd}"
	#       FileUtils.rm_rf(Dir.pwd)
	#     else:
	#       entries.each do |e|
	#         if File.directory? e:
	#           remove_leaves(e)




	#
	# def delete(self):
	#   raise SecurityError "cannot delete directories"
	# FileUtils.remove_dir(to_path, True)


class Dir(Directory):
	pass


class xdir(Directory):
	pass


# class Number < Numeric
#
# def not None(self):
#   return True
#
# def None.test(self):
#   "None.test OK"
#
# def None.+ x(self):
#   x

# def None.to_s:
#  ""
#  #"None"
#
@extension
class xdict(dict):
	def clone(self):
		import copy
		return copy.copy(self)

	# return copy.deepcopy(self)

	# filter ==  x.select{|z|z>1}

	# CAREFUL! MESSES with rails etc!!
	# alias_method :orig_index, :[]


	# Careful hash.map returns an Array, not a map as expected
	# Therefore we need a new method:
	# {a:1,b:2}.map_values{|x|x*3} => {a:3,b:6}


	# if not normed here too!!: DANGER: NOT surjective
	# def []= x,y # NORM only through getter:
	#   super[x.to_sym]=y
	#
	# def __index__(self,x):
	#   if not x: return
	#   if isinstance(x, symtable.Symbol): return orig_index(x)  or  orig_index(str(x))
	#   if isinstance(x,str): return orig_index(x)
	#   # yay! todo: eqls {a:b}=={:a=>b}=={"a"=>b} !!
	#   return orig_index(x)

	def contains(self, key):
		return self.keys().contains(key)


class Class:
	def wrap(self):
		return str(self)  # TODO!?


# WOW YAY WORKS!!!
# ONLY VIA EXPLICIT CONSTRUCTOR!
# NOOOO!! BAAAD! isinstance(my_xlist,list) FALSE !!

from functools import partial  # for method_missing


@extension
class xlist(list):
	def unique(xs):
		return xlist(set(xs))

	#
	# def list(xs):
	# 	for x in xs:
	# 		print(x)
	# 	return xs

	def uniq(xs):
		return xlist(set(xs))

	def unique(xs):
		return xlist(set(xs))

	def add(self, x):
		self.insert(len(self), x)

	def method_missing(xs, name, *args, **kwargs):  # [2.1,4.8].int=[2,5]
		if len(xs) == 0: return None
		try:
			method = getattr(xs.first(), name)
		except:
			# if str(name) in globals():
			method = globals()[str(name)]  # .method(m)
		if not callable(method):
			properties = xlist(map(lambda x: getattr(x, name), xs))
			return xlist(zip(xs, properties))
		# return properties
		return xlist(map(lambda x: method(args, kwargs), xs))  # method bound to x

	def pick(xs):
		return xs[randint(len(xs))]

	def __getattr__(self, name):
		if str(name) in globals():
			method = globals()[str(name)]  # .method(m)
			try:
				return method(self)
			except:
				xlist(map(method, self))
		return self.method_missing(name)

	# return partial(self.method_missing, name)

	def select(xs, func):  # VS MAP!!
		# return [x for x in xs if func(x)]
		return filter(func, xs)

	def map(xs, func):
		return xlist(map(func, xs))

	def last(xs):
		return xs[-1]

	def first(xs):
		return xs[0]

	def fold(self, x, fun):
		if not callable(fun):
			fun, x = x, fun
		return self.reduce(fun, self, x)

	def row(xs, n):
		return xs[int(n) - 1]

	def column(xs, n):
		if isinstance(xs[0], str):
			return xlist(map(lambda row: xstr(row).word(n + 1), xs))
		if isinstance(xs[0], list):
			return xlist(map(lambda row: row[n], xs))
		raise Exception("column of %s undefined" % type(xs[0]))

	# c=self[n]

	def length(self):
		return len(self)

	def clone(self):
		import copy
		return copy.copy(self)

	# return copy.deepcopy(self)

	def flatten(self):
		from itertools import chain
		return list(chain.from_iterable(self))

	def __gt__(self, other):
		if not isinstance(other, list): other = [other]
		return list.__gt__(self, other)

	def __lt__(self, other):
		if not isinstance(other, list): other = [other]
		return list.__lt__(self, other)

	# TypeError: unorderable types: int() < list() fucking python 3
	# def __cmp__(self, other):
	# 	if not isinstance(other, list): other = [other]
	# 	return list.__cmp__(self, other)

	def __sub__(self, other):  # xlist-[1]-[2] minus
		if not hasattr(other, '__iter__'): other = [other]
		return xlist(i for i in self if i not in other)

	def __rsub__(self, other):  # [1]-xlist-[2] ok!
		return xlist(i for i in other if i not in self)

	def c(self):
		return xlist(map(str.c, self).join(", "))  # leave [] which is not compatible with C

	def wrap(self):
		# map(wrap).join(", ") # leave [] which is not compatible with C
		return "rb_ary_new3(#{size}/*size*', #{wraps})"  # values

	def wraps(self):
		return xlist(map(lambda x: x.wrap, self).join(", "))  # leave [] which is not compatible with C

	def values(self):
		return xlist(map(lambda x: x.value, self).join(", "))  # leave [] which is not compatible with C

	def contains_a(self, type):
		for a in self:
			if isinstance(a, type): return True
		return False

	def drop(self, x):
		return self.reject(x)

	def to_s(self):
		return self.join(", ")

	# ifdef $auto_map:
	# def method_missing(method, *args, block):
	#   if args.count==0: return self.map (lambda x: x.send(method ))
	#   if args.count>0: return self.map (lambda x: x.send(method, args) )
	# super method, *args, block

	# def matches(item):
	#   contains item
	#
	# remove: confusing!!
	def matches(self, regex):
		for i in self.flatten():
			m = regex.match(i.gsub(r'([^\w])', "\\\\\\1"))  # escape_token(i))
			if m:
				return m
		return False

	def And(self, x):
		if not isinstance(x, list): self + [x]
		return self + x

	def plus(self, x):
		if not isinstance(x, list): self + [x]
		return self + x

	# EVIL!!
	# not def(self):
	#   None? not or

	# def = x  unexpected '=':
	#  is x
	#
	# def grep(x):
	#  select{|y|y.to_s.match(x)}
	#
	def names(self):
		return xlist(map(str, self))

	def rest(self, index=1):
		return self[index:]

	def fix_int(self, i):
		if str(i) == "middle": i = self.count() / 2
		if isinstance(i, Numeric): return i - 1
		i = xstr(i).parse_integer()
		return i - 1

	def character(self, nr):
		return self.item(nr)

	def item(self, nr):  # -1 AppleScript style ! BUT list[0] !
		return self[xlist(self).fix_int(nr)]

	def word(self, nr):  # -1 AppleScript style ! BUT list[0] !):
		return self[xlist(self).fix_int(nr)]

	def invert(self):  # ! Self modifying !
		self.reverse()
		return self

	def get(self, x):
		return self[self.index(x)]

	# def row(self, n):
	#     return self.at(n)

	def has(self, x):
		return self.index(x)

	def contains(self, x):
		ok = self.index(x)
		if ok:
			return self.at(self.index(x))
		else:
			return False

		# def to_s:
		#  "["+join(", ")+"]"
		#


		# class TrueClass:
		#   not def(self):
		#     False


class FalseClass:
	# not def(self):
	#   True

	def wrap(self):
		return self

	def c(self):
		return self


@extension
class xstr(str):
	# @staticmethod
	# def invert(self):
	#     r=reversed(self) #iterator!
	#     return "".join(r)

	def invert(self):
		r = reversed(self)  # iterator!
		self = "".join(r)
		return self

	def inverse(self):
		r = reversed(self)  # iterator!
		return "".join(r)

	def reverse(self):
		r = reversed(self)  # iterator!
		return "".join(r)

	def to_i(self):
		return int(self)

	# to_i=property(to_i1,to_i1)

	def quoted(self):
		return "%s" % self

	# def c(self):
	#     return self.quoted()

	# def id(self):
	#     return "id(%s)" % self
	#
	# def wrap(self):
	#     return "s(%s)" % self

	# def value(self):
	#     return self  # variable
	# quoted

	# def name(self):
	#     return self

	def number(self):
		return int(self)

	def is_in(self, ary):
		return ary.has(self)

	def cut_to(self, pattern):
		return self.sub(0, self.indexOf(pattern))

	def matches(self, regex):
		if isinstance(regex, list):
			for x in regex:
				if re.match(x):
					return x
		else:
			return re.match(regex)
		return False

	def strip_newline(self):
		return self.strip()  # .sub(r';$', '')

	def join(self, x):
		return self + x

	# def < x:
	#   i=x.is_a?Numeric
	#   if i:
	#     return int(self)<x
	#
	#   super.< x
	#
	def starts_with(self, x):
		# puts "WARNING: start_with? missspelled as starts_with?"
		if isinstance(x, list):
			for y in x:
				if self.startswith(y): return y
		return self.startswith(x)

	# def show(self, x=None):
	#     print(x or self)
	#     return x or self

	def contains(self, *things):
		for t in flatten(things):
			if self.index(t): return True
		return False

	def fix_int(self, i):
		if str(i) == "middle": i = self.count / 2
		if isinstance(i, int): return i - 1
		i = xstr(i).parse_integer()
		return i - 1

	def sentence(self, i):
		i = self.fix_int(i)
		return self.split(r'[\.\?\!\;]')[i]

	def paragraph(self, i):
		i = self.fix_int(i)
		return self.split("\n")[i]

	def word(self, i):
		i = self.fix_int(i)
		replaced = self.replace("\t", " ").replace("  ", " ").replace("\t", " ").replace("  ", " ")  # WTF
		words = replaced.split(" ")
		if i >= len(words): return self  # be gentle
		return words[i]

	def item(self, i):
		return self.word(i)

	def char(self, i):
		return self.character(i)

	def character(self, i):
		i = self.fix_int(i)
		return self[i - 1:i]

	def flip(self):
		return self.split(" ").reverse.join(" ")

	def plus(self, x):
		return self + x

	def _and(self, x):
		return self + x

	def add(self, x):
		return self + x

	def offset(self, x):
		return self.index(x)

	def __sub__(self, x):
		return self.gsub(x, "")

	# self[0:self.index(x)-1]+self[self.index(x)+x.length:-1]

	def synsets(self, param):
		pass

	def is_noun(self):  # expensive!):
		# Sequel::InvalidOperation Invalid argument used for IS operator
		return self.synsets('noun') or self.gsub(r's$', "").synsets('noun')  # except False

	def is_verb(self):
		return self.synsets('verb') or self.gsub(r's$', "").synsets('verb')

	def is_a(className):
		className = className.lower()
		if className == "quote": return True
		return className == "string"

	def is_adverb(self):
		return self.synsets('adverb')

	def is_adjective(self):
		return self.synsets('adjective')

	def examples(self):
		return xlist(self.synsets.flatten.map('hyponyms').flatten().map('words').flatten.uniq.map('to_s'))

	# def not_(self):
	#   return None or not
	def lowercase(self):
		return self.lower()

	# def replace(self,param, param1):
	#   pass

	def replaceAll(self, pattern, string):
		return re.sub(pattern, string, self)

	def shift(self, n=1):
		n.times(self=self.replaceAll(r'^.', ""))

	# self[n:-1]

	def replace_numerals(self):
		x = self
		x = x.replace(r'([a-z])-([a-z])', "\\1+\\2")  # WHOOOT???
		x = x.replace("last", "-1")  # index trick
		# x = x.replace("last", "0")  # index trick
		x = x.replace("first", "1")  # index trick

		x = x.replace("tenth", "10")
		x = x.replace("ninth", "9")
		x = x.replace("eighth", "8")
		x = x.replace("seventh", "7")
		x = x.replace("sixth", "6")
		x = x.replace("fifth", "5")
		x = x.replace("fourth", "4")
		x = x.replace("third", "3")
		x = x.replace("second", "2")
		x = x.replace("first", "1")
		x = x.replace("zero", "0")

		x = x.replace("4th", "4")
		x = x.replace("3rd", "3")
		x = x.replace("2nd", "2")
		x = x.replace("1st", "1")
		x = x.replace("(\d+)th", "\\1")
		x = x.replace("(\d+)rd", "\\1")
		x = x.replace("(\d+)nd", "\\1")
		x = x.replace("(\d+)st", "\\1")

		x = x.replace("a couple of", "2")
		x = x.replace("a dozen", "12")
		x = x.replace("ten", "10")
		x = x.replace("twenty", "20")
		x = x.replace("thirty", "30")
		x = x.replace("forty", "40")
		x = x.replace("fifty", "50")
		x = x.replace("sixty", "60")
		x = x.replace("seventy", "70")
		x = x.replace("eighty", "80")
		x = x.replace("ninety", "90")

		x = x.replace("ten", "10")
		x = x.replace("eleven", "11")
		x = x.replace("twelve", "12")
		x = x.replace("thirteen", "13")
		x = x.replace("fourteen", "14")
		x = x.replace("fifteen", "15")
		x = x.replace("sixteen", "16")
		x = x.replace("seventeen", "17")
		x = x.replace("eighteen", "18")
		x = x.replace("nineteen", "19")

		x = x.replace("ten", "10")
		x = x.replace("nine", "9")
		x = x.replace("eight", "8")
		x = x.replace("seven", "7")
		x = x.replace("six", "6")
		x = x.replace("five", "5")
		x = x.replace("four", "4")
		x = x.replace("three", "3")
		x = x.replace("two", "2")
		x = x.replace("one", "1")
		x = x.replace("dozen", "12")
		x = x.replace("couple", "2")

		# x = x.replace("½", "+.5");
		x = x.replace("½", "+1/2.0");
		x = x.replace("⅓", "+1/3.0");
		x = x.replace("⅔", "+2/3.0");
		x = x.replace("¼", "+.25");
		x = x.replace("¼", "+1/4.0");
		x = x.replace("¾", "+3/4.0");
		x = x.replace("⅕", "+1/5.0");
		x = x.replace("⅖", "+2/5.0");
		x = x.replace("⅗", "+3/5.0");
		x = x.replace("⅘", "+4/5.0");
		x = x.replace("⅙", "+1/6.0");
		x = x.replace("⅚", "+5/6.0");
		x = x.replace("⅛", "+1/8.0");
		x = x.replace("⅜", "+3/8.0");
		x = x.replace("⅝", "+5/8.0");
		x = x.replace("⅞", "+7/8.0");

		x = x.replace(" hundred thousand", " 100000")
		x = x.replace(" hundred", " 100")
		x = x.replace(" thousand", " 1000")
		x = x.replace(" million", " 1000000")
		x = x.replace(" billion", " 1000000000")
		x = x.replace("hundred thousand", "*100000")
		x = x.replace("hundred ", "*100")
		x = x.replace("thousand ", "*1000")
		x = x.replace("million ", "*1000000")
		x = x.replace("billion ", "*1000000000")
		return x

	def parse_integer(self):
		n = self.replace_numerals()
		i = int(n)  # except 666
		# i = int(eval(str(self)))  # except 666
		return i

	def parse_number(self):
		x = self.replace_numerals()
		try:
			x = float(x)
		except:
			x = eval(x)  # !! danger!
		if x == 0: return "0"  # ZERO
		return x

	# def __sub__(self, other): # []= MISSING in python!!
	#     x="abc"
	#     >>> x[2]='a'
	#     TypeError: 'str' object does not support item assignment WTF

	def reverse(self):
		# return self[slice(start=None,stop=None,step=-1)]
		return self[::-1]  # very pythonic,  It works by doing [begin:end:step]

	# a slower approach is ''.join(reversed(s))

	@staticmethod
	def reverse_string(str):
		return xstr(str).reverse()


class xchar(unicode):  # unicode: multiple bases have instance lay-out conflict
	def __coerce__(self, other):
		if isinstance(other, int):
			other = chr(other)
		# if isinstance(other,str):
		#     other=chr(other)
		return type(other)(self), other


# class Fixnum Float
# class Numeric:
# @Extension(int)
@extension
class xint(int):
	# operator.truth(obj)
	#     Return True if obj is true, and False otherwise. This is equivalent to using the bool constructor.
	# operator.is_(a, b)
	#     Return a is b. Tests object identity.
	# operator.is_not(a, b)
	#     Return a is not b. Tests object identity.

	def __coerce__(self, other):
		return int(other)

	# def __cmp__(self, other):
	# 	if isinstance(other, list): return list.__cmp__([self], other)

	def c(self):  # unwrap, for optimization):
		return str(self)  # "NUM2INT(#{self.to_s})"

	def value(self):
		return self

	def wrap(self):
		return "INT2NUM(#{self.to_s})"

	def number(self):
		return self

	def _and(self, x):
		return self + x

	def plus(self, x):
		return self + x

	def minus(self, x):
		return self - x

	def times(self, x):
		if callable(x):
			return [x() for i in xrange(self)]
		else:
			return self * x

	def times_do(self, fun):
		x = None
		for i in range(0, self):
			x = fun()
		return x

	def less(self, x):
		if isinstance(x, str): return self < int(x)
		return super.__lt__(x)

	def is_blank(self):
		return False

	def is_a(self, clazz):
		className = str(clazz).lower()
		if className == "number": return True
		if className == "real": return True
		if className == "float": return True
		# int = ALL : Fixnum = small int  AND :. Bignum = big : 2 ** (1.size * 8 - 2)
		if isinstance(self, int) and className == "integer": return True  # todo move
		if isinstance(self, int) and className == "int": return True
		if className == str(self).lower(): return True  # KINDA
		if self.isa(clazz): return True
		return False

	def add(self, x):
		return self + x

	def increase(self, by=1):
		return self + by  # Can't change the value of numeric self!!

	def decrease(self, by=1):
		return self - by  # Can't change the value of numeric self!!

	def bigger(self, x):
		return self > x

	def smaller(self, x):
		return self < x

	def to_the_power_of(self, x):
		return self**x

	def to_the(self, x):
		return self**x

	def logarithm(self):
		return math.log(self)

	def e(self):
		return math.exp(self)

	def exponential(self):
		return math.exp(self)

	def sine(self):
		return math.sin(self)

	def cosine(self):
		return math.cos(self)

	def root(self):
		return math.sqrt(self)

	def power(self, x):
		return self**x

	def square(self):
		return self * self

	# todo: use ^^
	def squared(self):
		return self * self


class Numeric(xint):
	pass


class Integer(xint):
	@classmethod
	def __eq__(self, other):
		if other == int: return True
		if other == xint: return True
		if other == Integer: return True
		return False


@extension
class xfloat(float):
	def to_i(self):
		return int(self)

	def c(self):  # unwrap, for optimization):
		return str(self)  # "NUM2INT(#{self.to_s})"

	def value(self):
		return self

	def number(self):
		return self

	def _and(self, x):
		return self + x

	def add(self, x):
		return self + x

	def plus(self, x):
		return self + x

	def minus(self, x):
		return self - x

	def times(self, x):
		return self * x

	def less(self, x):
		if isinstance(x, str): return self < int(x)
		return super.__lt__(x)

	def is_blank(self):
		return False

	def is_a(self, clazz):
		className = str(clazz).lower()
		if className == "number": return True
		if className == "real": return True
		if className == "float": return True
		# int = ALL : Fixnum = small int  AND :. Bignum = big : 2 ** (1.size * 8 - 2)
		if isinstance(self, int) and className == "integer": return True  # todo move
		if isinstance(self, int) and className == "int": return True
		if className == str(self).lower(): return True  # KINDA
		if self.isa(clazz): return True
		return False

	def increase(self, by=1):
		return self + by  # Can't change the value of numeric self!!

	def decrease(self, by=1):
		return self - by  # Can't change the value of numeric self!!

	def bigger(self, x):
		return self > x

	def smaller(self, x):
		return self < x

	def is_bigger(self, x):
		return self > x

	def is_smaller(self, x):
		return self < x

	def to_the_power_of(self, x):
		return self**x

	def to_the(self, x):
		return self**x

	def logarithm(self):
		return math.log(self)

	def e(self):
		return math.exp(self)

	def exponential(self):
		return math.exp(self)

	def sine(self):
		return math.sin(self)

	def cosine(self):
		return math.cos(self)

	def root(self):
		return math.sqrt(self)

	def power(self, x):
		return self**x

	def square(self):
		return self * self

	# todo: use ^^
	def squared(self):
		return self * self


# if self==false: return True
# if self==True: return false
# class Enumerator

@extension  # DANGER?
class xobject:
	def __init__(selfy):
		selfy.self = selfy

	def value(self):
		return self

	def number(self):
		return False

	# not def(self):
	# return    False

	def throw(self, x):
		raise x

	def type(self):
		return self.__class__

	def kind(self):
		return self.__class__

	def log(*x):
		print(x)

	def debug(*x):
		print(x)

	def is_a(self, clazz):
		if self is clazz: return True
		try:
			ok = isinstance(self, clazz)
			if ok: return True
		except Exception as e:
			print(e)

		className = str(clazz).lower()
		if className == str(self).lower(): return True  # KINDA

		if self.is_(clazz): return True
		return False

	def is_(self, x):
		if not x and not self: return True
		if x == self: return True
		if x is self: return True
		if str(x).lower() == str(self).lower(): return True  # KINDA
		if isinstance(self, list) and self.length == 1 and x.is_(self[0]): return True
		if isinstance(x, list) and x.length == 1 and self.is_(x[0]):  return True
		return False


def load(file):
	return open(file, 'rt').read()


def load_binary(file):
	return open(file, 'rb').read()


def read(file):
	return open(file, 'rt').read()



def readlines(source):
	# print("open(source).readlines()")
	return list(map(str.strip, open(source).readlines()))

# def readlines(file):
# 	return open(file, 'rt').readlines()


def writelines(file, xs):
	open(file, 'wt').write("\n".join(xs))

def read_binary(file):
	return open(file, 'rb').read()


def dump(o, file="dump.bin"):
	pickle.dump(o, open(file, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
	print("saved to '" + file + "'")


save = dump
write = dump  # ok for plain bytes too++


def write_direct(data, file):
	open(file, 'wb').write(data)


def load_pickle(file_name="dump.bin"):
	return pickle.load(open(file_name, 'rb'))


def unpickle(file_name="dump.bin"):
	return pickle.load(open(file_name, 'rb'))


def undump(file_name="dump.bin"):
	return pickle.load(open(file_name, 'rb'))


def restore(file_name="dump.bin"):
	return pickle.load(open(file_name, 'rb'))


def run(cmd):
	os.system(cmd)


def exists(file):
	os.path.exists(file)


# class Encoding:
#     pass


def find_in_module(module, match="", recursive=True):  # all
	if isinstance(module, str):
		module = sys.modules[module]
	for name, obj in inspect.getmembers(module):
		# if inspect.isclass(obj):
		if match in name:
			print(obj)
		if inspect.ismodule(obj) and recursive and obj != module:
			if module.__name__ in obj.__name__:
				# print("SUBMODULE: %s"%obj)
				find_in_module(obj, match)


def find_class(match=""):  # all
	import sys, inspect
	for module in sys.modules.keys():  # sys.modules[module] #by key
		for name, obj in inspect.getmembers(sys.modules[module]):
			if inspect.isclass(obj):
				if match in str(obj):
					print(obj)


# @extension
# class Math:
# WOOOT? just
# import math as Math
# def __getattr__(self, attr):
#     import sys
#     import math
#  # ruby method_missing !
#     import inspect
#     for name, obj in inspect.getmembers(sys.modules['math']):
#         if name==attr: return obj
#     return False

print("extensions loaded")
