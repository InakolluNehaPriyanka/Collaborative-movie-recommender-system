import numpy
from contextlib import redirect_stdout
# Dynamic Programming implementation of LCS problem
 
def lcs(X , Y):
    # find the length of the strings
    m = len(X)
    n = len(Y)
 
    # declaring the array for storing the dp values
    L = [[None]*(n+1) for i in range(m+1)]
 
    """Following steps build L[m+1][n+1] in bottom up fashion
    Note: L[i][j] contains length of LCS of X[0..i-1]
    and Y[0..j-1]"""
    for i in range(m+1):
        for j in range(n+1):
            if i == 0 or j == 0 :
                L[i][j] = 0
            elif X[i-1] == Y[j-1]:
                L[i][j] = L[i-1][j-1]+1
            else:
                if L[i-1][j] > L[i][j-1]:
                    L[i][j]=L[i-1][j]
                else:
                    L[i][j]=L[i][j-1]
                
 
    # L[m][n] contains the length of LCS of X[0..n-1] & Y[0..m-1]
    return L[m][n]
#end of function lcs
 
 
# Driver program to test the above function
A=numpy.array(['def','bcq','q','bfi','giq','i','ip','efi','i','ir','gq','gq','fl','io','i','fo','bfgq','i','i','io','bcfmq','bir','iq','bcg','f','f','b','biq','bcfg','il','iqr','h','boq','fi','cei','io','i','pq','bgp','f','f','f','iq','iq','fi','i','fi','h','fo','bcopr','ior','il','bq','biq','gioq','gi','i','i','i','i','i','bcp','ef','i','fi','fo','f','boq','for','fo','dem','fgj','bf','bfi','h','gi','iq','cei','bq','bfr','fo','bcp','fo','p','f','i','i','fo','kp','foq','efm','bgo','fi','ef','defm','bpq','ci','iq','dem','giq','bcdp','de','dem','f','f','iq','i','f','fp','bcfr','fo','cel','il','dl','h','f','bcq','bcq','h','fg','bpr','f','f','in','io','i','bgi','bq','giloq','gl','io','ceim','ior','i','inpq','i','il','ef','ef','ce','cejp','cem','m','bq','bpq','il','bq','bcl','i','fi','cef','fp','f','f','mo','gq','ir','f','nq','i','bo','i','fl','bcpq','i','i','f','f','df','fio','fp','bciopr','bcfo','bc','p','bpqr','b','i','p','ir','bcopr','gi','bpq','bcfp','oq','bfm','bgi','bir','df','ir','in','i','i','fg','bpq','i','io','q','ir','bcf','fo','fo','fo','','fp','ir','cdpq','bio','f','fim','bc','fr','i','io','imr','i','fo','lo','q','','fo','i','bcp','iq','i','ef','bq','bcp','bcp','bcp','bcp','bcfg','bf','b','b','bfpr','fil','io','f','gip','df','bor','f','ef','biq','biqr','io','bce','fg','f','bp','f','bcpq','io','bcg','fo','fo','bcfp','ip','ef','bnpq','ef','i','b','pq','bq','bc','i','io','f','ipq','bcpr','il','bgq','fo','io','io','i','io','i','ilo','bq','il','io','fo','i','ior','i','q','im','f','nq','i','gi','f','bq','i','i','bpq','gik','bq','f','gknq','i','ce','i','io','gnq','eijl','g','i','ioq','fgin','bio','be','iq','fi','i','ir','fmo','h','f','nq','bq','n','ilq','bir','gin','bnoq','giq','i','cq','giq','nq','bgn','fl','gq','fiq','f','bi','i','f','fln','bp','i','fi','gi','fi','giq','bq','bnq','','fm','blp','fo','cpq','inq','i','bcpq','q','h','gq','bfm','b','f','il','il','fl','fl','fl','ilq','ilo','fl','bclp','bel','il','fl','efl','i','','bcp','flo','fil','ef','f','bcfo','f','i','bf','ce','f','bf','i','f','fno','f','fg','b','bcep','bcf','ef','f','foq','bcgi','de','bcn','q','f','dfq','fi','f','fjop','f','','f','ef','ei','ei','dem','efm','dem','io','def','eijp','','f','bdepqr','i','f','ip','fr','bc','dem','f','p','bf','','','','','','','','','p','','','','','bcp','bcp','fmo','b','b','i','b','bf','cei','i','i','i','i','io','c','i','ceo','q','i','i','i','','ir','bcj','dem','pr','i','f','ef','fo','nq','flq','fi','fg','ior','kn','mo','fo','fo','kl','koq','floq','bc','il','ln','fl','cf','il','f','bclor','i','ce','dem','flr','i','gi','nq','i','i','i','i','bi','cr','fio','nq','fo','bir','f','fio','i','cl','cr','ir','fi','fi','f','kn','bci','i','ir','i','c','io','o','n','i','fo','i','i','dem','ef','b','bc','demo','im','gio','flo','bq','g','ej','ior','bq','','p','io','bc','i','','im','ijq','i','cefjop','i','bc','','fl','q','bclq','','boq','i','','fq','bq','pq','noq','f','bcg','fp','bp','f','fo','iq','io','gq','ei','f','b','in','dem','','bp','iq','nq','r','ilo','lq','delm','blq','fil','bl','el','clr','mo','lnq','fl','m','i','oq','noq','f','lm','gkln','i','f','i','q','p','i','i','iq','i','','ce','ef','dem','de','ei','i','gi','fm','fm','bior','i','i','h','l','bcpq','f','i','i','i','ir','gik','io','h','h','','ir','fo','giq','i','bir','fi','gkq','knq','cfi','gkq','kq','i','fnq','i','','io','f','i','bpq','','','b','','pq','','','kq','','','i','h','iq','bc','bc','','nq','f','bq','bq','bi','fr','f','bq','ir','kpq','fio','i','o','fi','ilq','i','i','i','f','h','fo','il','io','mo','l','i','i','fo','f','i','fi','il','io','fi','io','ilq','f','fr','bcio','f','f','fi','io','f','ei','io','fp','il','io','fio','fo','io','f','i','io','fi','fo','fo','io','iq','iq','bq','ir','fl','f','fo','boq','fo','i','boq','bq','i','gq','bcejp','f','h','pq','b','p','bq','i','fl','fo','fo','h','','cel','bcnp','gknq','bpq','i','q','','fio','i','','fio','b','f','fo','il','flo','i','fo','fo','fil','lq','fil','fl','e','f','f','io','ef','fo','bp','b','i','q','f','bcgq','bir','f','fn','bgi','i','bi','bin','b','h','ce','h','hl','io','','i','f','f','cdefj','fo','i','gkq','fl','bpq','c','bcq','ce','bc','bnq','bcpq','eij','b','q','fmo','fo','i','b','q','bi','bq','efi','ef','g','f','io','hi','kq','bo','h','i','iq','f','f','binoq','fi','hl','','f','q','','cef','i','f','i','fo','i','i','fo','o','f','o','fo','i','io','bf','co','efn','bqr','i','bf','i','i','h','','oq','i','i','o','bc','ir','efj','f','ef','q','i','bcp','i','i','i','f','fgnq','io','fi','io','io','f','f','i','i','gi','bq','f','ginq','i','bpq','biq','o','cp','i','io','','i','ci','pq','f','i','i','ef','bcq','pq','ef','i','i','ir','fio','i','fo','iq','f','fi','i','q','fir','fnoq','de','co','fo','io','i','cej','f','fi','h','io','i','f','i','f','i','i','i','iq','fi','fm','o','i','m','de','f','fr','i','h','iq','q','bpq','b','nq','iq','i','il','bcq','q','q','il','q','q','fo','dem','io','g','fq','cdefm','i','f','ef','f','f','f','f','f','f','efn','i','i','ip','f','i','i','i','g','fi','bcq','f','ei','bcq','i','i','bq','nq','i','h','f','o','biq','fi','b','fo','f','f','ce','ef','ce','bc','f','fj','fmo','q','i','fo','fo','nq','f','fi','i','q','f','fo','f','io','f','f','i','f','i','','f','ci','f','ce','fi','i','ei','gk','hr','de','f','i','giq','f','f','fo','q','fi','i','bcdej','q','de','fm','f','g','i','gq','h','io','il','b','b','boq','q','cdem','f','f','f','f','o','i','f','i','fio','in','fo','fi','fgi','bcq','gi','i','i','iq','bfmp','i','i','i','f','fio','c','i','f','io','f','im','gi','i','or','q','ci','i','hl','ino','h','q','io','cej','i','im','i','io','bfgi','bgq','f','h','h','gq','i','i','i','i','i','i','i','g','or','im','p','fo','gi','','i','np','io','bi','f','i','e','fi','f','f','i','i','fi','i','fl','io','iq','o','ir','f','f','f','bf','bf','f','bf','h','fr','fi','g','bf','i','fo','gi','i','i','gi','i','i','fi','gq','f','ei','h','f','fmo','fir','i','f','q','giq','f','pq','io','ino','giq','i','bp','o','q','f','defo','i','i','b','i','i','f','gi','i','b','iq','f','bi','h','i','f','g','i','fi','i','bco','dp','fi','f','','b','g','f','','q','f','b','f','i','q','f','i','i','i','fo','fo','i','i','o','i','i','i','i','i','i','fo','f','f','fi','iq','pq','q','i','bg','im','ce','g','i','i','f','in','i','fmo','f','f','o','io','f','ej','cejp','h','fi','fi','io','fm','io','ilo','f','f','b','bg','f','i','h','fi','i','i','fi','fi','knq','bcq','io','i','i','h','i','i','f','f','i','iq','i','i','i','i','i','i','h','i','fi','i','i','efj','f','i','fi','o','i','i','i','i','i','i','i','i','f','i','f','i','o','i','i','f','or','b','i','f','f','b','h','b','bi','hi','d','i','i','iq','f','b','a','f','i','f','fo','h','o','i','i','i','ce','i','fo','p','o','i','i','g','i','i','bq','o','i','i','i','i','q','io','i','i','fi','f','i','ilo','b','f','de','i','cp','de','b','b','be','bp','i','i','bp','f','i','kq','ir','fio','fi','fi','i','f','io','i','q','i','bi','o','f','io','f','i','gi','i','io','i','i','efn','i','f','i','i','il','co','q','fn','gi','','','fi','fi','fmo','m','fi','f','gio','f','i','fi','i','i','i','ce','de','q','fp','ei','fo','fi','k','q','bgi','f','cef','i','h','bio','bf','ior','i','fo','i','f','f','biq','f','o','f','i','fg','h','f','f','f','r','flo','ce','f','gi','i','f','iq','f','g','f','i','f','i','co','i','i','i','gi','','fo','q','bc','i','fo','ioq','f','i','r','i','ce','i','i','fg','i','i','f','i','i','ce','i','i','i','f','io','i','h','bq','q','f','i','b','nq','il','ce','biq','fi','i','biq','g','hi','h','o','i','i','i','f','fo','fi','f','i','i','','r','i','j','i','i','q','o','n','k','i','i','h','bgi','','fi','f','i','i','i','f','h','bq','bpq','bi','gq','i','f','q','i','i','gkq','f','g','i','cei','f','bgo','f','o','bi','fi','bcej','i','o','bg','i','f','q','f','i','q','1q','fi','fi','i','h','i','fi','r','fi','i','i','i','i','gi','i','i','h','i','i','f','i','bi','fgi','i','fh','i','iq','o','i','f','fo','fo','bi','i','f','i','i','io','ir','f','i','i','i','f','gin','fq','i','i','bq','i','i','i','i','i','oq','io','f','i','i'
])
for p in range(1,1683):
    for q in range (1,1683):
        
            X = A[p]
            Y = A[q]
            dr=len(X)*len(Y)
            if dr!=0:
                sm=lcs(X,Y)/dr
            else:
                sm=0
            with open('lcs_output.txt', 'a') as f:
                with redirect_stdout(f):
                    print(p,".....",q,"...",sm)
    
            
 
