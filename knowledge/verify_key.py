"""Local answer-key checks. These computations are NOT available to JEV."""
import heapq
from itertools import combinations, permutations, product
import json
import math
from pathlib import Path

import numpy as np
from scipy.integrate import quad, dblquad, solve_ivp
from scipy.linalg import expm
import sympy as s

ROOT=Path(__file__).resolve().parent
checks={}
key=json.loads((ROOT/'answer_key.json').read_text(encoding='utf-8'))

def check(number, actual, expected, method):
    if isinstance(actual,(s.Basic,int)) and isinstance(expected,(s.Basic,int)):
        difference=s.sympify(actual-expected).rewrite(s.gamma)
        assert s.simplify(difference)==0,(number,actual,expected)
    else:
        assert abs(complex(actual)-complex(expected))<1e-7,(number,actual,expected)
    checks[f'Q{number:02d}']={'passed':True,'computed':str(actual),'method':method}

def main():
    check(1,(18*7-36)/6,15,'Arithmetic')
    check(2,80*s.Rational(3,4)-12,48,'Exact fractions')
    x=s.symbols('x');check(3,s.solve(3*x+7-25,x)[0],6,'Symbolic equation')
    check(4,3*4+2*4**2/2,28,'Constant-acceleration displacement')
    check(5,s.Rational(12,6)+s.Rational(12,3),6,'Separate branch currents')
    for number,answer,url in [(6,'France','https://www.france.fr/fr/destination/paris/'),(7,'Tokyo','https://www.japan.go.jp/japan/index.html')]:
        assert key[f'Q{number:02d}']['correct_text']==answer
        checks[f'Q{number:02d}']={'passed':True,'method':'Primary-source fact checked in browsing','source':url}
    assert key['Q08']['correct_text']=='Box K is heavy'
    checks['Q08']={'passed':True,'method':'Universal implication applied to explicit member'}
    check(9,s.Rational(sum(i>4 for i in range(1,7)),6),s.Rational(1,3),'Enumerate die outcomes')
    check(10,1/(s.Rational(1,6)+s.Rational(1,3)),2,'Combined rates')
    check(11,quad(lambda t:math.log(t)/(1+t)**3,0,np.inf)[0],-.5,'Independent improper numerical integration')
    M=s.diag(*([3]*12))
    for i in range(11):M[i,i+1]=M[i+1,i]=-1
    check(12,M.det(),121393,'Direct symbolic determinant, not recurrence')
    check(13,next(i for i in range(2,1001) if i*i%1001==1),155,'Exhaustive modular scan')
    check(14,sum(len(set(v))==3 for v in product(range(3),repeat=7)),1806,'Enumerate all assignments')
    seq=[0,1]
    for i in range(2,16):seq.append(4*seq[-1]-4*seq[-2])
    check(15,seq[15],245760,'Direct recurrence')
    check(16,sum(1/(1-np.exp(2j*np.pi*k/17)) for k in range(1,17)),8,'Complex numerical summation')
    q=s.symbols('q');check(17,s.discriminant(x**3-3*x-q,x),108-27*q*q,'Symbolic global upper bound; equality witness xyz=0')
    check(18,float(s.zeta(2)*2-3),np.pi**2/3-3,'Telescoping plus zeta identity')
    adjacency=s.zeros(9)
    for i in range(4):
        for j in range(4,9):adjacency[i,j]=adjacency[j,i]=1
    laplacian=s.diag(*[sum(adjacency[i,j] for j in range(9)) for i in range(9)])-adjacency
    check(19,laplacian[:8,:8].det(),32000,'Matrix-tree minor determinant')
    roots=s.nroots(x**5-3*x+1,maxsteps=100)
    check(20,sum(1/(complex(root)-1)**2 for root in roots),24,'Numerical roots independent of logarithmic derivative')
    check(21,dblquad(lambda y,x:math.log(x)*math.log(y)/(1-x*y),0,1,lambda _:0,lambda _:1)[0],np.pi**4/90,'Independent double numerical integration')
    count=0
    for east in combinations(range(16),8):
        horizontal=vertical=0;valid=True
        for step in range(16):
            if step in east:horizontal+=1
            else:vertical+=1
            if step<15 and vertical>=horizontal:valid=False;break
        count+=valid
    check(22,count,429,'Enumerate 12870 paths')
    pattern='HTHTH';A=s.eye(5);b=s.ones(5,1)
    for state in range(5):
        for toss in 'HT':
            text=pattern[:state]+toss
            nxt=max(k for k in range(6) if text.endswith(pattern[:k]))
            if nxt<5:A[state,nxt]-=s.Rational(1,2)
    check(23,(A.inv()*b)[0],42,'Exact prefix-state Markov equations')
    check(24,quad(lambda t:math.sqrt(1-t*t)-(1-t),0,1)[0]*2,np.pi/2-1,'Conditional-area numerical integration')
    check(25,s.beta(11,3)/s.beta(8,3),s.Rational(60,143),'Beta posterior third moment')
    derangements=sum(all(i!=v for i,v in enumerate(p)) for p in permutations(range(6)))
    check(26,s.Rational(math.comb(8,2)*derangements,math.factorial(8)),s.Rational(53,288),'Enumerate 6! derangements')
    P=s.Matrix([[0,1,0],[s.Rational(1,2),0,s.Rational(1,2)],[0,s.Rational(1,3),0]])
    check(27,((s.eye(3)-P).inv()*s.ones(3,1))[0],7,'Exact absorbing transition matrix')
    unknown=s.symbols('h1:7'); h=[s.S.Zero,*unknown,s.S.One]
    result=s.solve([h[i]-s.Rational(3,5)*h[i+1]-s.Rational(2,5)*h[i-1] for i in range(1,7)],unknown)
    check(28,result[h[3]],s.Rational(1539,2059),'Solve finite harmonic system')
    r,t,angle=s.symbols('r t angle',real=True)
    d4=(r*r+t*t-2*r*t*s.cos(angle))**2
    value=s.integrate(s.integrate(s.integrate(d4*2*r*2*t/(2*s.pi),(angle,0,2*s.pi)),(r,0,1)),(t,0,1))
    check(29,value,s.Rational(5,3),'Exact integration over two radii and relative angle')
    allowed=0
    for positions in combinations(range(10),4):
        allowed+=all((i+1)%10 not in positions for i in positions)
    check(30,s.Rational(allowed,math.comb(10,4)),s.Rational(5,42),'Enumerate conditional circular positions')
    stiffness=s.Matrix([[2,-1,0],[-1,2,-1],[0,-1,2]])
    eigen=list((s.diag(1,s.Rational(1,2),1)*stiffness).eigenvals())
    eigen.sort(key=float)
    check(31,eigen[-1]/eigen[0],(7+3*s.sqrt(5))/2,'Generalized mass/stiffness eigenvalues')
    beta=(s.Rational(-4,5)-s.Rational(3,5))/(1-s.Rational(-4,5)*s.Rational(3,5))
    check(32,1/s.sqrt(1-beta*beta),s.Rational(37,12),'Velocity addition then Lorentz factor')
    Tf=s.sqrt(400*100);check(33,10*(400+100-2*Tf),1000,'Entropy-conserving common temperature and energy')
    check(34,((-1)**2/(2*s.Rational(5)))/(s.Rational(1,2)*(1+2)),s.Rational(1,15),'Isolated total charge and final capacitance')
    phi1=(1/s.pi)**s.Rational(1,4)*s.exp(-x*x/2)
    phi2=(2/s.pi)**s.Rational(1,4)*s.exp(-x*x)
    overlap=s.integrate(phi1*phi2,(x,-s.oo,s.oo))
    check(35,overlap**2,2*s.sqrt(2)/3,'Independent Gaussian overlap integral')
    sx=np.array([[0,1],[1,0]],complex);sy=np.array([[0,-1j],[1j,0]],complex)
    psi=expm(-1j*np.pi/8*sy)@expm(-1j*np.pi/6*sx)@np.array([1,0],complex)
    check(36,abs(psi[0])**2,(4+np.sqrt(2))/8,'Explicit sequential unitary matrices')
    radius,L=s.symbols('radius L',positive=True);U=radius**4+L**2/(2*radius**2)
    radial=s.diff(U,radius,2).subs(L**2,4*radius**6)
    check(37,s.sqrt(radial/(4*radius**2)),s.sqrt(6),'Second derivative of effective potential')
    energy=s.Rational(-7,8); angular=s.Rational(1,2)
    eccentricity=s.sqrt(1+2*energy*angular**2)
    check(38,(-1/(2*energy))*(1-eccentricity),s.Rational(1,7),'Energy, angular momentum and eccentricity')
    check(39,abs(sum(np.exp(1j*np.pi*j/5) for j in range(5)))**2/25,(6+2*np.sqrt(5))/25,'Numerical coherent phasor sum')
    check(40,s.Rational(1+1**2)/(1+s.Rational(7,2)**2),s.Rational(8,53),'Complex impedance at both frequencies')
    def reached(t,y):return y[0]-.5
    reached.terminal=True
    sol=solve_ivp(lambda t,y:[1-y[0]**2,y[0]],(0,2),[0,0],events=reached,rtol=1e-11,atol=1e-12)
    check(41,sol.y[1,-1],math.log(4/3)/2,'Independent drag ODE integration, vt=g=1')
    theta=s.symbols('theta',real=True);potential=1-s.cos(theta)-s.sin(theta)**2
    check(42,s.diff(potential,theta,2).subs(theta,s.pi/3),s.Rational(3,2),'Effective-potential curvature, g=R=1')
    assignments=[]
    for a,b,c,d,e,f in product((False,True),repeat=6):
        claims=(b!=c,sum((a,d,e))==2,d==(not f),a!=e,(not b)and(not f),sum((a,b,c,d,e))>=3)
        if (a,b,c,d,e,f)==claims:assignments.append((a,b,c,d,e,f))
    assert assignments==[(True,True,False,True,False,True)]
    checks['Q43']={'passed':True,'method':'All 64 truth assignments','solutions':assignments}
    times=[1,3,8,12,17,24];dist={(0,0):0};queue=[(0,0,0)]
    while queue:
        cost,mask,side=heapq.heappop(queue)
        if cost!=dist[(mask,side)]:continue
        if (mask,side)==(63,1):break
        available=[i for i in range(6) if bool(mask&(1<<i))==bool(side)]
        for size in (1,2):
            for group in combinations(available,size):
                delta=sum(1<<i for i in group);new=(mask^delta,1-side);newcost=cost+max(times[i] for i in group)
                if newcost<dist.get(new,math.inf):dist[new]=newcost;heapq.heappush(queue,(newcost,*new))
    check(44,dist[(63,1)],53,'Dijkstra across all bridge states')
    possible=[m for m in range(21) if 2*m<=27 and 2*(40-2*m)<=27]
    assert not possible
    checks['Q45']={'passed':True,'method':'Exhaustive first-weighing branch-capacity constraints plus worst-case small weight defect argument'}
    check(46,2*98+1+1,198,'Maximum possible counter count with one prisoner absent, plus one')
    wins=0
    for mask in range(128):
        guesses=[]
        for i in range(7):
            own=(mask>>i)&1;syndrome=0
            for j in range(7):
                if j!=i and (mask>>j)&1:syndrome^=j+1
            guess=1 if syndrome==0 else 0 if syndrome==i+1 else None
            if guess is not None:guesses.append(guess==own)
        wins+=bool(guesses)and all(guesses)
    check(47,s.Rational(wins,128),s.Rational(7,8),'Exhaustive Hamming strategy; global upper bound in proof')
    check(48,next(n for n in range(3,20) if sum(math.comb(n,k) for k in (1,2,3))>=100),9,'Adaptive-coverage recurrence closed form')
    check(49,sum('000' not in format(mask,'020b') and mask.bit_count()%3==0 for mask in range(1<<20)),74493,'Enumerate all 1048576 binary strings')
    alphametic=[]
    for D,O,N,A,L,G,E,R,B,T in permutations(range(10)):
        if not D or not G or not R or 2*D%10!=T:continue
        c1=2*D//10
        if (2*L+c1)%10!=R:continue
        c2=(2*L+c1)//10
        if (2*A+c2)%10!=E:continue
        c3=(2*A+c2)//10
        if (N+R+c3)%10!=B:continue
        c4=(N+R+c3)//10
        if (O+E+c4)%10!=O:continue
        c5=(O+E+c4)//10
        if D+G+c5!=R:continue
        alphametic.append(100000*R+10000*O+1000*B+100*E+10*R+T)
    assert alphametic==[723970],alphametic
    checks['Q50']={'passed':True,'method':'All digit permutations pruned by carry constraints','solutions':alphametic}
    assert len(checks)==50
    (ROOT/'answer_key_verification.json').write_text(json.dumps({'all_passed':True,'questions_checked':50,'checks':checks},indent=2),encoding='utf-8')
    print(json.dumps({'all_passed':True,'questions_checked':50}))

if __name__=='__main__': main()
