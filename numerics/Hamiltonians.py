import numpy as np
from interactions import *

def nearest_nb_Hamiltonian(b,h,a,chain_length,interaction,Delta,d):
    latticeA = np.zeros((2,chain_length//2+chain_length%2))
    latticeB = np.zeros((2,chain_length//2))
    for n in range(latticeA.shape[1]):
        latticeA[:,n] = n*np.array([1,0])*a
    for n in range(latticeB.shape[1]):
        latticeB[:,n] = n*np.array([1,0])*a + b * np.array([1,0])+h*np.array([0,1])
    dmat = np.zeros((chain_length,chain_length))
    ind = np.arange(chain_length,dtype=np.int16)
    inter_1 = interaction(latticeB[:,0]-latticeA[:,0],d)
    inter_2 = interaction(latticeB[:,0]-latticeA[:,1],d)
    dmat[ind[:-1:2],ind[1::2]]= inter_1
    dmat[ind[1::2],ind[:-1:2]]= inter_1
    dmat[ind[1:-1:2],ind[2::2]] = inter_2
    dmat[ind[2::2],ind[1:-1:2]] = inter_2
    dmat[ind[::2],ind[::2]] = Delta
    dmat[ind[1::2],ind[1::2]]= -Delta
    # print(inter_1,inter_2,b)
    return dmat#

from qutip import enr_destroy, enr_fock, enr_identity, enr_state_dictionaries, tensor, Qobj, enr_thermal_dm
from scipy.special import factorial, binom
from scipy.sparse import coo_matrix, csr_matrix


def add_one(state,ex):
    
    if np.sum(state)==ex:
        state[-1] = 0
        state[:-1] = add_one(state[:-1],ex)
    else:
        state[-1] += 1
    return state



def project_indices(excitations,chain_length):
    state = np.zeros(chain_length,dtype=np.int16)
    indices = []
    if np.sum(state)==excitations:
        return 0
    for i in range(1,int(binom(chain_length+excitations,excitations))):
        state = add_one(state,excitations)
        if np.sum(state)==excitations:
            # print(i, state)
            indices.append(i)
    # discard = int(factorial(excitations+chain_length-1)//(factorial(excitations-1)*factorial(chain_length)))
    # return discard
    return indices

def projector(indices,*operators):
        # discard = int(factorial(excitations+chain_length-1)//(factorial(excitations-1)*factorial(chain_length)))
        new_qops = []
        for operator in operators:
            # if type(operator)==complex:
            #     return enr_destroy([excit_phonons+1]*chain_length,excit_phonons)
            operator_full = operator.to("CSR").data_as("csr_matrix")
            operator_new = operator_full[indices,:][:,indices]
            # operator_new = operator_full[indices:,indices:]
            # .tocoo()
            # data, coords = operator_full.data, operator_full.coords
            # coords = np.stack((coords[0]-discard,coords[1]-discard))
            # # print(coords)
            # # print(discard)
            # # coords -= discard
            # operator_new = coo_matrix((data,coords),shape=[operator_full.shape[0]-int(discard)]*2)#.tocsr()
            # print(type(operator_new))
            if len(operators)==1:
                return Qobj(operator_new)
            new_qops.append(Qobj(operator_new))
        return new_qops

def project_vec(indices,*operators):
        # discard = int(factorial(excitations+chain_length-1)//(factorial(excitations-1)*factorial(chain_length)))
        new_qops = []
        for operator in operators:
            operator_full = operator.full()[indices]#operator.to("CSR").data_as("csr_matrix")
            # operator_new = operator_full[indices]
            # print(operator_new)
            if len(operators)==1:
                return Qobj(operator_full)
            new_qops.append(Qobj(operator_full))
        return new_qops

def operators_zero(chain_length,excit_spin=1):
   
    indices_spin = project_indices(excit_spin,chain_length)
    
    dims_spin = [excit_spin+1]*chain_length
    
    destroy_spin = enr_destroy(dims_spin,excitations=excit_spin)
   
    identity_spin = projector(indices_spin,enr_identity(dims_spin,excitations=excit_spin))
    
    project_spin =[]
    spin_flips = []
    
    
    
    for i in range(len(destroy_spin)):
        
        project_spin.append(projector(indices_spin,destroy_spin[i].dag()*destroy_spin[i]))
        
       
       
        if i<len(destroy_spin)-1:
            op_sp_flip = destroy_spin[i].dag()*destroy_spin[i+1]
            spin_flips.append(projector(indices_spin,op_sp_flip))
     
    phonon_flips, project_phonon, identity_phonon= [0]*3
    return phonon_flips, spin_flips, project_phonon, project_spin, identity_phonon ,identity_spin, project_spin

def operators(chain_length,excit_phonons,excit_spin=1):
    if excit_phonons==0:
        return operators_zero(chain_length,excit_spin)
    indices = project_indices(excit_phonons,chain_length)
    indices_spin = project_indices(excit_spin,chain_length)
    dims_phonon = [excit_phonons+1]*chain_length
    dims_spin = [excit_spin+1]*chain_length
    destroy_phonon = enr_destroy(dims_phonon,excitations=excit_phonons)
    destroy_spin = enr_destroy(dims_spin,excitations=excit_spin)
    identity_phonon = projector(indices,enr_identity(dims_phonon,excitations=excit_phonons))
    identity_spin = projector(indices_spin,enr_identity(dims_spin,excitations=excit_spin))
    project_phonon = []
    project_spin =[]
    spin_flips = []
    phonon_flips = []
    eval_projector_spin = []
    eval_projector_phonon = []
    for i in range(len(destroy_phonon)):
        project_phonon.append(projector(indices,destroy_phonon[i].dag()*destroy_phonon[i]))
        project_spin.append(projector(indices_spin,destroy_spin[i].dag()*destroy_spin[i]))
        eval_projector_spin.append(tensor(project_spin[i],identity_phonon))
        # print(project_phonon[i])
        eval_projector_phonon.append(tensor(identity_spin,project_phonon[i]))
        if i<len(destroy_phonon)-1:
            op_sp_flip = destroy_spin[i].dag()*destroy_spin[i+1]
            op_ph_flip = destroy_phonon[i].dag()*destroy_phonon[i+1]
            spin_flips.append(projector(indices_spin,op_sp_flip))
            phonon_flips.append(projector(indices,op_ph_flip))

    return phonon_flips, spin_flips, project_phonon, project_spin, identity_phonon ,identity_spin, eval_projector_spin+eval_projector_phonon#[::-1]


def operators_full(chain_length,excit_phonons,excit_spin=1):
    dims_phonon = [excit_phonons+1]*chain_length
    dims_spin = [excit_spin+1]*chain_length
    destroy_phonon = enr_destroy(dims_phonon,excitations=excit_phonons)
    destroy_spin = enr_destroy(dims_spin,excitations=excit_spin)
    identity_phonon = enr_identity(dims_phonon,excitations=excit_phonons)
    identity_spin = enr_identity(dims_spin,excitations=excit_spin)
    project_phonon = []
    project_spin =[]
    spin_flips = []
    phonon_flips = []
    eval_projector_spin = []
    eval_projector_phonon = []
    for i in range(len(destroy_phonon)):
        project_phonon.append(destroy_phonon[i].dag()*destroy_phonon[i])
        project_spin.append(destroy_spin[i].dag()*destroy_spin[i])
        eval_projector_spin.append(tensor(project_spin[i],identity_phonon))
        eval_projector_phonon.append(tensor(identity_spin,project_phonon[i]))
        if i<len(destroy_phonon)-1:
            op_sp_flip = destroy_spin[i].dag()*destroy_spin[i+1]
            op_ph_flip = destroy_phonon[i].dag()*destroy_phonon[i+1]
            spin_flips.append(op_sp_flip)
            phonon_flips.append(op_ph_flip)

    return phonon_flips, spin_flips, project_phonon, project_spin, identity_phonon ,identity_spin, eval_projector_spin+eval_projector_phonon


class system():
    def __init__(self,chain_length,excit_phonons,interaction_sp=dip_dip,interaction_spph=dip_derivative_full,excit_spin=1,a=1,d=np.array([np.sqrt(1/3),np.sqrt(2/3)]),ph_scale=1/50,sp_scale=1,full=True):
        self.chain_length = chain_length
        self.excit_phonons = excit_phonons
        self.excit_spin = excit_spin
        self.interaction_sp = interaction_sp
        self.interaction_spph = interaction_spph
        self.a = a
        self.d = d
        self.ph_scale = ph_scale
        self.sp_scale = sp_scale
        if full:
            self.phonon_flips, self.spin_flips, self.project_phonon, self.project_spin, self.identity_phonon, self.identity_spin, self.projectors_eval = operators_full(chain_length,excit_phonons,excit_spin)
        else:
            self.phonon_flips, self.spin_flips, self.project_phonon, self.project_spin, self.identity_phonon, self.identity_spin, self.projectors_eval = operators(chain_length,excit_phonons,excit_spin)
        self.eval_functions = []
        for i in range(len(self.projectors_eval)):
            self.eval_functions.append(lambda t,rho, i=i : (self.projectors_eval[i]*rho).tr())

            
    param_trajectory = None
    
    # def projector(self,operator):
    #     discard = factorial(self.excit_spin+self.chain_length-2)//(factorial(self.excit_spin-1)*factorial(self.chain_length-1))
    #     operator_full = operator.to("CSR").data_as("csr_matrix").tocoo()
    #     data, coords = operator_full.data, operator_full.coords
    #     coords = np.stack((coords[0]-discard,coords[1]-discard))
    #     # print(coords)
    #     # print(discard)
    #     # coords -= discard
    #     operator_new = coo_matrix((data,coords),shape=[operator_full.shape[0]-int(discard)]*2)#.tocsr()
    #     # print(type(operator_new))
    #     return Qobj(operator_new)
    
    def Hamiltonian_spin_only(self,Delta,interaction1,interaction2):
        Ham_spin = 0
        # print(interaction1,interaction2,"spin")
        for i in range(self.chain_length//2):
            Ham_spin += Delta * self.project_spin[2*i]- Delta * self.project_spin[2*i+1]
            op = interaction1* self.spin_flips[2*i]
            if  i<self.chain_length//2+self.chain_length%2-1:
                op2 = interaction2* self.spin_flips[2*i+1]
                Ham_spin += op2 + op2.dag()
            Ham_spin += op+op.dag()
        # return 0
        return Ham_spin

    def Hamiltonian_spin(self,Delta,interaction1,interaction2):
        Ham_spin = 0
        # print(interaction1,interaction2,"spin")
        for i in range(self.chain_length//2):
            Ham_spin += Delta * tensor(self.project_spin[2*i],self.identity_phonon) - Delta * tensor(self.project_spin[2*i+1],self.identity_phonon)
            op = interaction1* tensor(self.spin_flips[2*i],self.identity_phonon)
            if  i<self.chain_length//2+self.chain_length%2-1:
                op2 = interaction2* tensor(self.spin_flips[2*i+1],self.identity_phonon)
                Ham_spin += op2 + op2.dag()
            Ham_spin += op+op.dag()
        # return 0
        return Ham_spin

    def Hamiltonian_spin_phonon(self,interaction1,interaction2):
        Ham_ps= 0
        # print(interaction1,interaction2,"phonon")
        for i in range(self.chain_length//2):
            op_ph = interaction1 * (self.phonon_flips[2*i]+ self.phonon_flips[2*i].dag()- self.project_phonon[2*i]- self.project_phonon[2*i+1])
            
            Ham_ps += tensor(self.spin_flips[2*i],op_ph)

            if i<self.chain_length//2+self.chain_length%2-1:

                op_ph2 = interaction2 * (self.phonon_flips[2*i+1]+ self.phonon_flips[2*i+1].dag()- self.project_phonon[2*i+1]- self.project_phonon[2*(i+1)]) 
                
                Ham_ps += tensor(self.spin_flips[2*i+1],op_ph2)

        return Ham_ps + Ham_ps.dag()
    
    def Hamiltonian_phonon_only(self,b,h):
        latticeA = np.array([[0,0],[self.a,0]])
        latticeB = np.array([b,h])
        dist1 = latticeA[0]-latticeB
        dist2 = latticeA[1]-latticeB
        phonon_direction = 0
        inter_ph1 = dip_derivative_full(dist1,phonon_direction,phonon_direction,self.d)
        inter_ph2 = dip_derivative_full(dist2,phonon_direction,phonon_direction,self.d)
        Ham_ps= 0
        # print(interaction1,interaction2,"phonon")
        for i in range(self.chain_length//2):
            op_ph =  inter_ph1 *(self.phonon_flips[2*i]+ self.phonon_flips[2*i].dag()- self.project_phonon[2*i]- self.project_phonon[2*i+1])
            # op_ph =  
            Ham_ps += op_ph
            
            if i<self.chain_length//2+self.chain_length%2-1:

                op_ph2 = inter_ph2 * (self.phonon_flips[2*i+1]+ self.phonon_flips[2*i+1].dag()- self.project_phonon[2*i+1]- self.project_phonon[2*(i+1)]) 
                
                Ham_ps += op_ph2

        return Ham_ps #+ Ham_ps.dag()
         
    def Hamiltonian_spin_time(self,t):
        b, h, Delta, d = self.param_trajectory(t)
        latticeA = np.array([[0,0],[self.a,0]])
        latticeB = np.array([b,h])
        dist1 = latticeA[0]-latticeB
        dist2 = latticeA[1]-latticeB
        inter1 = dip_dip(dist1,d)
        inter2 = dip_dip(dist2,d)
        
        # return 0
        return self.Hamiltonian_spin_only(Delta,inter1,inter2) 

    def Hamiltonian_time(self,t):
        b, h, Delta, d = self.param_trajectory(t)
        latticeA = np.array([[0,0],[self.a,0]])
        latticeB = np.array([b,h])
        dist1 = latticeA[0]-latticeB
        dist2 = latticeA[1]-latticeB
        inter1 = dip_dip(dist1,d)
        inter2 = dip_dip(dist2,d)
        phonon_direction = 0
        inter_ph1 = dip_derivative_full(dist1,phonon_direction,phonon_direction,d)
        inter_ph2 = dip_derivative_full(dist2,phonon_direction,phonon_direction,d)
        # 
        return  self.sp_scale * self.Hamiltonian_spin(Delta,inter1,inter2) +self.ph_scale * self.Hamiltonian_spin_phonon(inter_ph1,inter_ph2)
    
    
    def Hamiltonian(self,b,h,Delta=0):
        latticeA = np.array([[0,0],[self.a,0]])
        latticeB = np.array([b,h])
        dist1 = latticeA[0]-latticeB
        dist2 = latticeA[1]-latticeB
        inter1 = dip_dip(dist1,self.d)
        inter2 = dip_dip(dist2,self.d)
        phonon_direction = 0
        inter_ph1 = dip_derivative_full(dist1,phonon_direction,phonon_direction,self.d)
        inter_ph2 = dip_derivative_full(dist2,phonon_direction,phonon_direction,self.d)
        # 
        # print(inter_ph1,inter_ph2,self.ph_scale,"phonon")
        # inter_ph1 = 1
        # inter_ph2 = 1
        return self.sp_scale * self.Hamiltonian_spin(Delta,inter1,inter2) +self.ph_scale *  self.Hamiltonian_spin_phonon(inter_ph1,inter_ph2)
    
    def Hamiltonian_spin_adjusted(self,b,h,Delta=0):
        latticeA = np.array([[0,0],[self.a,0]])
        latticeB = np.array([b,h])
        dist1 = latticeA[0]-latticeB
        dist2 = latticeA[1]-latticeB
        inter1 = dip_dip(dist1,self.d)
        inter2 = dip_dip(dist2,self.d)
        phonon_direction = 0
        inter_ph1 = dip_derivative_full(dist1,phonon_direction,phonon_direction,self.d)
        inter_ph2 = dip_derivative_full(dist2,phonon_direction,phonon_direction,self.d)
        ph_therm = Qobj(csr_matrix(np.ones(int(factorial(self.chain_length+self.excit_phonons-1)/(factorial(self.excit_phonons)*factorial(self.chain_length-1)))))).dag().unit()
        i = 0

        op_ph =  inter_ph1 *(self.phonon_flips[2*i]+ self.phonon_flips[2*i].dag()- self.project_phonon[2*i]- self.project_phonon[2*i+1])
 
        op_ph2 = inter_ph2 * (self.phonon_flips[2*i+1]+ self.phonon_flips[2*i+1].dag()- self.project_phonon[2*i+1]- self.project_phonon[2*(i+1)]) 

        add1 = ph_therm.dag() * op_ph * ph_therm
        add2 = ph_therm.dag() * op_ph2 * ph_therm
        print(add1,add2,"phonon",inter1,inter2,"spin")
        inter1 = self.sp_scale*inter1+self.ph_scale*add1
        inter2 = self.sp_scale*inter2+self.ph_scale*add2
        return self.sp_scale * self.Hamiltonian_spin_only(Delta,inter1,inter2)
    
    def Hamiltonian_spinner(self,b,h,Delta=0):
        latticeA = np.array([[0,0],[self.a,0]])
        latticeB = np.array([b,h])
        dist1 = latticeA[0]-latticeB
        dist2 = latticeA[1]-latticeB
        inter1 = dip_dip(dist1,self.d)
        inter2 = dip_dip(dist2,self.d)
        
        return self.Hamiltonian_spin_only(Delta,inter1,inter2) 



from qutip import tensor, basis, destroy, create, qeye, sesolve, sigmap, sigmam, fidelity, qzero, expect, ptrace


def full_operators(chain_length,boson_number):
    unity_spin = tensor([qeye(2)]*chain_length)
    annihilation_phonon = []
    phonon_projectors = []
    for i in range(chain_length):
        op_list = []
        op_proj = []
        for j in range(chain_length):
            if i == j:
                op_list.append(destroy(boson_number))
                op_proj.append(create(boson_number)*destroy(boson_number))
            else:
                op_list.append(qeye(boson_number))
                op_proj.append(qeye(boson_number))
        annihilation_phonon.append(tensor(op_list))
        phonon_projectors.append(tensor([qeye(2)]*chain_length+op_proj))
    unity_phonon = tensor([qeye(boson_number) for i in range(chain_length)])
    sigmas_p = []
    sigmas_m = []
    onsite_projectors_spin = []
    projectors_eval = []
    for i in range(chain_length):
        op_list = []
        op_list_id = []
        for j in range(chain_length):
            if i == j:
                op_list.append(sigmap())
                op_list_id.append(sigmap()*sigmam())
            else:
                op_list.append(qeye(2))
                op_list_id.append(qeye(2))
        sigmas_p.append(tensor(op_list))
        sigmas_m.append(tensor(op_list).dag())
        onsite_projectors_spin.append(tensor(op_list_id))
        projectors_eval.append(tensor(tensor(op_list_id),unity_phonon))
    projectors_eval = projectors_eval + phonon_projectors
    return annihilation_phonon, sigmas_m, sigmas_p, phonon_projectors, onsite_projectors_spin, unity_phonon, unity_spin, projectors_eval

class full_dynamics():
    def __init__(self,chain_length,boson_number,a=1,d=np.array([np.sqrt(1/3),np.sqrt(2/3)]),ph_scale=1/50):
        self.chain_length = chain_length
        self.boson_number = boson_number
        self.a = a
        self.d = d
        self.ph_scale = ph_scale
        self.annihilation_phonon, self.sigmas_m, self.sigmas_p, self.phonon_projectors, self.onsite_projectors_spin, self.unity_phonon, self.unity_spin, self.projectors_eval = full_operators(chain_length,boson_number=boson_number)

    
    
        self.latticeA = np.zeros((2,chain_length//2+chain_length%2))
        self.latticeB = np.zeros((2,chain_length//2))
        for n in range(self.latticeA.shape[1]):
            self.latticeA[:,n] = n*np.array([1,0])*a
        for n in range(self.latticeB.shape[1]):
            self.latticeB[:,n] = n*np.array([1,0])*a + a/2*np.array([1,0])
    
    def Hamiltonian_1Dph(self,Delta,b,h,phonon_direction=0):
        latticeA = np.array([[0,0],[self.a,0]])
        latticeB = np.array([b,h])
        dist1 = latticeA[0]-latticeB
        dist2 = latticeA[1]-latticeB
        inter1 = dip_dip(dist1,self.d)
        inter2 = dip_dip(dist2,self.d)
        inter_ph1 = dip_derivative_full(dist1,phonon_direction,phonon_direction,self.d)
        inter_ph2 = dip_derivative_full(dist2,phonon_direction,phonon_direction,self.d)
        # self.Hamiltonian_spin_only(Delta,inter1,inter2) + self.ph_scale * 
        # inter_ph1 = 1
        # inter_ph2 = 1
        return self.Hamiltonian_spin_phonon(inter_ph1,inter_ph2)


    def Hamiltonian_spin_only(self,Delta,interaction1,interaction2):
        Ham_spin = 0
        # print(interaction1,interaction2,"spin")
        for i in range(self.chain_length//2):
            Ham_spin += Delta * tensor(self.onsite_projectors_spin[2*i],self.unity_phonon) - Delta * tensor(self.onsite_projectors_spin[2*i+1],self.unity_phonon)
            op = interaction1* tensor(self.sigmas_p[i*2],self.unity_phonon)* tensor(self.sigmas_m[i*2+1],self.unity_phonon)
            if  i<self.chain_length//2+self.chain_length%2-1:
                op2 = interaction2* tensor(self.sigmas_p[i*2+1],self.unity_phonon) * tensor(self.sigmas_m[(i+1)*2],self.unity_phonon)
                Ham_spin += op2 + op2.dag()
            Ham_spin += op+op.dag()
        # return 0
        return Ham_spin

    def Hamiltonian_spin_phonon(self, interaction1, interaction2):
        Ham_ps= 0
        # print(interaction1,interaction2,"phonon")
        for i in range(self.chain_length//2):
            op_ph = interaction1 * (self.annihilation_phonon[2*i].dag()*self.annihilation_phonon[2*i+1] + self.annihilation_phonon[2*i+1].dag()*self.annihilation_phonon[2*i]-self.annihilation_phonon[2*i].dag()*self.annihilation_phonon[2*i]-self.annihilation_phonon[2*i+1].dag()*self.annihilation_phonon[2*i+1])
            Ham_ps += tensor(self.sigmas_p[i*2]*self.sigmas_m[i*2+1],op_ph)
            # Ham_ps += tensor(self.sigmas_p[i*2+1]*self.sigmas_m[i*2],op_ph)
            # Ham_ps += tensor(self.unity_spin,op_ph)
            if i<self.chain_length//2+self.chain_length%2-1:
                op_ph2 = interaction2 * (self.annihilation_phonon[2*i+1].dag()*self.annihilation_phonon[2*(i+1)] + self.annihilation_phonon[2*(i+1)].dag()*self.annihilation_phonon[2*i+1]-self.annihilation_phonon[2*i+1].dag()*self.annihilation_phonon[2*i+1]-self.annihilation_phonon[2*(i+1)].dag()*self.annihilation_phonon[2*(i+1)])
                Ham_ps += tensor(self.sigmas_p[i*2+1]*self.sigmas_m[2*(i+1)],op_ph2)
                # Ham_ps += tensor(self.sigmas_p[2*(i+1)]*self.sigmas_m[i*2+1],op_ph2)
                # Ham_ps += tensor(self.unity_spin,op_ph2)
        # return 0
        return Ham_ps + Ham_ps.dag()
         

# H =Hamiltonian_1Dph(1,a/2,0,a,0,np.array([0,1]),1)

