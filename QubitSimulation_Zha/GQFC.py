def func(flux,Capa,Rlist,couplingXYQQDirect):
    import numpy as np
    from QubitEnergyFineSimulation import GroundedTransmonWithFloatingCoupler1V4 
    RlistNew = Rlist.copy()
    RCoupler = Rlist[1]/np.cos(np.pi*flux)
    RlistNew[1] = RCoupler
    para = [Capa,RlistNew]
    FTWC = GroundedTransmonWithFloatingCoupler1V4(para)
    el = FTWC.energyLevel
    couplerFrequency = el[FTWC.stateIndexList[3]]
    qubitFreuqency = (el[FTWC.stateIndexList[2]]+el[FTWC.stateIndexList[1]])/2
    qubit02Freuqency = (el[FTWC.stateIndexList[5]]+el[FTWC.stateIndexList[6]])/2
    coulingXYQCQ = -abs(el[FTWC.stateIndexList[2]]-el[FTWC.stateIndexList[1]])/2 if FTWC.couplingMinus else abs(el[FTWC.stateIndexList[2]]-el[FTWC.stateIndexList[1]])/2
    anharmonicity = (qubit02Freuqency-2*qubitFreuqency)
    coulingZZQCQ = abs(el[FTWC.stateIndexList[2]]+el[FTWC.stateIndexList[1]]-el[FTWC.stateIndexList[4]])
    deltaFreqency = (qubitFreuqency-couplerFrequency)
    couplingQC = np.sqrt((coulingXYQCQ-couplingXYQQDirect)*deltaFreqency)
    result = {
        'couplerFrequency':couplerFrequency,
        'qubitFreuqency':qubitFreuqency,
        'el':el[0:8],
        'coulingXYQCQ':coulingXYQCQ,
        'coulingZZQCQ':coulingZZQCQ,
        'anharmonicity':anharmonicity,
        'couplingQC':couplingQC,
        'couplerLeakage':FTWC.couplerLeakage,
    }
    return(result)
