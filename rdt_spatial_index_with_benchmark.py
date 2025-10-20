# ============================================================================
# RDT SPATIAL INDEX v4.1 — GPU + CPU HYBRID (AUTO-FALLBACK + CUDA TIMING)
# ============================================================================

!pip install numba --quiet
import numpy as np, time
from numba import njit, cuda, int32
import math

# ----------------------------------------------------------------------------
# Core RDT math
# ----------------------------------------------------------------------------
@njit(fastmath=True, inline='always')
def rdt_grid_size(n, alpha=1.5):
    if n <= 1: return 2
    d = max(2, int(np.log(n+1)**alpha))
    return min(d, 32)

@njit(fastmath=True, inline='always')
def point_to_cell(x, y, x0, y0, cw, ch, g):
    ix = int((x-x0)/cw); iy = int((y-y0)/ch)
    ix = max(0, min(g-1, ix)); iy = max(0, min(g-1, iy))
    return iy*g + ix

@njit(fastmath=True, inline='always')
def cell_to_bounds(cid, g, x0, y0, cw, ch):
    ix = cid % g; iy = cid // g
    return x0+ix*cw, y0+iy*ch, x0+(ix+1)*cw, y0+(iy+1)*ch

@njit(fastmath=True, inline='always')
def circle_box(cx, cy, r2, x0, y0, x1, y1):
    cx2 = max(x0, min(cx, x1)); cy2 = max(y0, min(cy, y1))
    dx, dy = cx - cx2, cy - cy2
    return dx*dx + dy*dy <= r2

# ----------------------------------------------------------------------------
# Tree construction (CPU)
# ----------------------------------------------------------------------------
@njit
def create_tree_arrays(max_nodes, max_pts_per_node):
    node_x0=np.zeros(max_nodes); node_y0=np.zeros(max_nodes)
    node_x1=np.zeros(max_nodes); node_y1=np.zeros(max_nodes)
    depth=np.zeros(max_nodes,int32); leaf=np.ones(max_nodes,np.bool_)
    count=np.zeros(max_nodes,int32); grid=np.zeros(max_nodes,int32)
    start=np.zeros(max_nodes,int32)
    child=np.full((max_nodes,1024),-1,int32)
    px=np.zeros(max_nodes*max_pts_per_node)
    py=np.zeros_like(px)
    return (node_x0,node_y0,node_x1,node_y1,depth,leaf,count,grid,start,child,px,py)

@njit
def init_root(arrs, x0,y0,x1,y1):
    nx0,ny0,nx1,ny1,depth,leaf,count,grid,start,child,px,py=arrs
    nx0[0],ny0[0],nx1[0],ny1[0]=x0,y0,x1,y1
    leaf[0]=True; depth[0]=0; start[0]=0
    return 1

@njit
def build_tree(px_in,py_in,arrs,nid,pid,alpha,max_leaf,max_depth):
    nx0,ny0,nx1,ny1,depth,leaf,count,grid,start,child,px,py=arrs
    n=len(px_in); max_s=len(px)
    start[0]=0
    for i in range(n):
        if i>=max_s: break
        px[i]=px_in[i]; py[i]=py_in[i]
    count[0]=n; pid=n
    stack=[0]
    while stack:
        node=stack.pop()
        if count[node]<=max_leaf or depth[node]>=max_depth: continue
        g=rdt_grid_size(count[node],alpha)
        grid[node]=g
        w=nx1[node]-nx0[node]; h=ny1[node]-ny0[node]
        cw, ch = w/g, h/g
        st=start[node]; cnt=count[node]
        cell_ct=np.zeros(g*g,int32); cellid=np.zeros(cnt,int32)
        for i in range(cnt):
            cid=point_to_cell(px[st+i],py[st+i],nx0[node],ny0[node],cw,ch,g)
            cellid[i]=cid; cell_ct[cid]+=1
        for cid in range(g*g):
            if cell_ct[cid]==0: continue
            if nid>=len(nx0): break
            cid_new=nid; nid+=1
            x0,y0,x1,y1=cell_to_bounds(cid,g,nx0[node],ny0[node],cw,ch)
            nx0[cid_new],ny0[cid_new],nx1[cid_new],ny1[cid_new]=x0,y0,x1,y1
            depth[cid_new]=depth[node]+1; leaf[cid_new]=True
            if pid+cell_ct[cid]>=max_s: break
            start[cid_new]=pid; count[cid_new]=0
            widx=pid
            for i in range(cnt):
                if cellid[i]==cid:
                    px[widx]=px[st+i]; py[widx]=py[st+i]
                    widx+=1; count[cid_new]+=1
            pid=widx; child[node,cid]=cid_new; stack.append(cid_new)
        leaf[node]=False; count[node]=0
    return nid,pid

# ----------------------------------------------------------------------------
# GPU kernel – same recursive logic, explicit stack
# ----------------------------------------------------------------------------
@cuda.jit
def gpu_query_rdt(qx,qy,r2,node_x0,node_y0,node_x1,node_y1,
                  leaf,count,grid,start,child,px,py,res):
    i = cuda.grid(1)
    if i>=qx.size: return
    cx,cy = qx[i],qy[i]
    stack = cuda.local.array(64,int32)
    sp=0; stack[sp]=0
    hits=0
    while sp>=0:
        n=stack[sp]; sp-=1
        if not circle_box(cx,cy,r2,node_x0[n],node_y0[n],node_x1[n],node_y1[n]):
            continue
        if leaf[n]:
            s=start[n]; c=count[n]
            for j in range(c):
                dx=px[s+j]-cx; dy=py[s+j]-cy
                if dx*dx+dy*dy<=r2: hits+=1
        else:
            g=grid[n]
            if g<=0: continue
            for cid in range(g*g):
                ch=child[n,cid]
                if ch>=0 and sp<63:
                    sp+=1; stack[sp]=ch
    res[i]=hits

# ----------------------------------------------------------------------------
# CPU fallback (same traversal)
# ----------------------------------------------------------------------------
@njit(fastmath=True)
def cpu_query(cx,cy,r2,arrs):
    nx0,ny0,nx1,ny1,depth,leaf,count,grid,start,child,px,py=arrs
    stack=[0]; hits=0
    while stack:
        n=stack.pop()
        if not circle_box(cx,cy,r2,nx0[n],ny0[n],nx1[n],ny1[n]): continue
        if leaf[n]:
            s=start[n]; c=count[n]
            for j in range(c):
                dx=px[s+j]-cx; dy=py[s+j]-cy
                if dx*dx+dy*dy<=r2: hits+=1
        else:
            g=grid[n]
            for cid in range(g*g):
                ch=child[n,cid]
                if ch>=0: stack.append(ch)
    return hits

# ----------------------------------------------------------------------------
# Unified wrapper
# ----------------------------------------------------------------------------
class RDTv41:
    def __init__(self,x0=0,y0=0,x1=1000,y1=1000,alpha=1.5,max_leaf=128,max_pts=1_000_000):
        self.arrs=create_tree_arrays(min(100000,max(1000,max_pts//max_leaf*10)),max_leaf*3)
        self.bounds=(x0,y0,x1,y1)
        self.alpha=alpha; self.max_leaf=max_leaf
        self.built=False; self.gpu_ready=False

    def build(self,pts):
        px=np.array([p[0] for p in pts]); py=np.array([p[1] for p in pts])
        self.nid=init_root(self.arrs,*self.bounds)
        self.nid,_=build_tree(px,py,self.arrs,self.nid,0,self.alpha,self.max_leaf,20)
        self.count=len(pts); self.built=True
        if cuda.is_available():
            nx0,ny0,nx1,ny1,depth,leaf,count,grid,start,child,px,py=self.arrs
            self.dnode_x0=cuda.to_device(nx0); self.dnode_y0=cuda.to_device(ny0)
            self.dnode_x1=cuda.to_device(nx1); self.dnode_y1=cuda.to_device(ny1)
            self.dleaf=cuda.to_device(leaf); self.dcount=cuda.to_device(count)
            self.dgrid=cuda.to_device(grid); self.dstart=cuda.to_device(start)
            self.dchild=cuda.to_device(child)
            self.dpx=cuda.to_device(px); self.dpy=cuda.to_device(py)
            self.gpu_ready=True
            print(f"✅ GPU tree uploaded ({self.count:,} pts, {self.nid:,} nodes)")
        else:
            print("⚠️ GPU unavailable — using CPU fallback")

    def query(self,queries,radius):
        qx=np.array([q[0] for q in queries]); qy=np.array([q[1] for q in queries])
        r2=radius*radius; res=np.zeros(len(qx),dtype=np.int32)

        if self.gpu_ready:
            dqx,dqy=cuda.to_device(qx),cuda.to_device(qy)
            dres=cuda.to_device(res)
            threads=256; blocks=(len(qx)+threads-1)//threads

            # --- precise CUDA timing ---
            start_evt=cuda.event(); end_evt=cuda.event()
            start_evt.record()
            gpu_query_rdt[blocks,threads](dqx,dqy,r2,
                self.dnode_x0,self.dnode_y0,self.dnode_x1,self.dnode_y1,
                self.dleaf,self.dcount,self.dgrid,self.dstart,self.dchild,
                self.dpx,self.dpy,dres)
            end_evt.record(); end_evt.synchronize()
            ms=cuda.event_elapsed_time(start_evt,end_evt)
            dres.copy_to_host(res)
            print(f"⏱️ GPU kernel time: {ms:.3f} ms")
        else:
            tb=time.perf_counter()
            for i in range(len(qx)):
                res[i]=cpu_query(qx[i],qy[i],r2,self.arrs)
            print(f"⏱️ CPU fallback time: {time.perf_counter()-tb:.4f}s")

        return res

# ----------------------------------------------------------------------------
# Benchmark demo
# ----------------------------------------------------------------------------
def benchmark_v41():
    from scipy.spatial import cKDTree
    sizes=[10_000,50_000,100_000,500_000]
    for n in sizes:
        print(f"\n{'='*80}\nDataset: {n:,} points\n{'='*80}")
        pts=[(np.random.uniform(0,1000),np.random.uniform(0,1000)) for _ in range(n)]
        queries=[(np.random.uniform(0,1000),np.random.uniform(0,1000)) for _ in range(100)]
        tree=cKDTree(np.array(pts))
        t0=time.perf_counter(); _=[tree.query_ball_point(q,50) for q in queries];
        print(f"cKDTree query: {time.perf_counter()-t0:.4f}s")

        rdt=RDTv41(max_pts=n); rdt.build(pts)
        t1=time.perf_counter(); _=rdt.query(queries,50)
        print(f"Total RDTv4.1 time: {time.perf_counter()-t1:.4f}s")

print("\n🔥 Compiling kernels (warm-up)...")
warm=[(float(i),float(i)) for i in range(100)]
rdt=RDTv41(max_pts=100); rdt.build(warm)
_ = rdt.query([(50,50)],10)
print("✅ Warm-up complete!\n")

benchmark_v41()
