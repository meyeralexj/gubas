def hou_config_read(filename):
    CFG=cfp.ConfigParser()
    CFG.read(filename)
    
    # get flags
    fahnestock_flag=CFG.getboolean("Initial Conditions","Fahnestock Input File Flag")
    C_flag=CFG.getboolean("Initial Conditions","B into A Euler Flag")
    Cc_flag=CFG.getboolean("Initial Conditions","A into N Euler Flag")
    integ=CFG.getint("Integration Settings", "Integrator Flag")
    Tgen=CFG.getint("Body Model Definitions","Inertia Integral Generation Flag")
    a_shape=CFG.getint("Body Model Definitions","Primary Shape Flag")
    b_shape=CFG.getint("Body Model Definitions","Secondary Shape Flag")
    
    # get gracity expansion values
    n=CFG.getint("Mutual Gravity Expansion Parameters","Gravity Expansion Truncation Order")
    nA=CFG.getint("Mutual Gravity Expansion Parameters","Primary Inertia Integral Truncation Order")
    nB=CFG.getint("Mutual Gravity Expansion Parameters","Secondary Inertia Integral Truncation Order")

    # get integrator settings
    t0=CFG.getfloat("Integration Settings","Start Time")
    tf=CFG.getfloat("Integration Settings","Final Time")
    h=CFG.getfloat("Integration Settings","Fixed Time Step")
    tol=CFG.getfloat("Integration Settings","Absolute Tolerance")
    
    # get shape settings - values converted to km
    aA=CFG.getfloat("Body Model Definitions","Primary Semi-Major Axis")
    bA=CFG.getfloat("Body Model Definitions","Primary Semi-Intermediate Axis")
    cA=CFG.getfloat("Body Model Definitions","Primary Semi-Minor Axis")
    aB=CFG.getfloat("Body Model Definitions","Secondary Semi-Major Axis")
    bB=CFG.getfloat("Body Model Definitions","Secondary Semi-Intermediate Axis")
    cB=CFG.getfloat("Body Model Definitions","Secondary Semi-Minor Axis")
    
    tet_fileA=CFG.get("Body Model Definitions","Primary Tetrahedron File")
    vert_fileA=CFG.get("Body Model Definitions","Primary Vertex File")
    tet_fileB=CFG.get("Body Model Definitions","Secondary Tetrahedron File")
    vert_fileB=CFG.get("Body Model Definitions","Secondary Vertex File")
    
    # get output settings
    postProcessing=CFG.getint("Output Settings", "Post Processing") #added by hagrusa, option to only output binaries
    out_freq=CFG.getfloat("Output Settings", "Fixed Output Frequency")
    out_time_name=CFG.get("Output Settings","Specified Time List Filename")
    case=CFG.get("Output Settings","Case Name")

    # get additional forces settings
    flyby_toggle = CFG.getint("Additional Forces and Perturbations","Flyby")
    sg_toggle = CFG.getint("Additional Forces and Perturbations","Solar Gravity")
    tt_toggle = CFG.getint("Additional Forces and Perturbations","Tidal Torque")
    helio_toggle = CFG.getint("Additional Forces and Perturbations","Heliocentric Orbit")
    Mplanet = CFG.getfloat("Additional Forces and Perturbations","Planetary Mass")
    a_hyp = CFG.getfloat("Additional Forces and Perturbations","Semimajor Axis")
    e_hyp = CFG.getfloat("Additional Forces and Perturbations","Eccentricity")
    i_hyp = CFG.getfloat("Additional Forces and Perturbations","Inclination")
    RAAN_hyp = CFG.getfloat("Additional Forces and Perturbations","RAAN")
    om_hyp = CFG.getfloat("Additional Forces and Perturbations","Argument of Periapsis")
    tau_hyp = CFG.getfloat("Additional Forces and Perturbations","Flyby Time")
    Msolar = CFG.getfloat("Additional Forces and Perturbations","Solar Mass")
    a_helio = CFG.getfloat("Additional Forces and Perturbations","Heliocentric Semimajor Axis")
    e_helio = CFG.getfloat("Additional Forces and Perturbations","Heliocentric Eccentricity")
    i_helio = CFG.getfloat("Additional Forces and Perturbations","Heliocentric Inclination")
    RAAN_helio = CFG.getfloat("Additional Forces and Perturbations","Heliocentric RAAN")
    om_helio = CFG.getfloat("Additional Forces and Perturbations","Heliocentric Argument of Periapsis")
    tau_helio = CFG.getfloat("Additional Forces and Perturbations","Time of periapsis passage")
    sol_rad = CFG.getfloat("Additional Forces and Perturbations","Solar Orbit Radius")
    au_def = CFG.getfloat("Additional Forces and Perturbations","AU Definition")/1000.
    love1 = CFG.getfloat("Additional Forces and Perturbations","Primary Love Number")
    love2 = CFG.getfloat("Additional Forces and Perturbations","Secondary Love Number")
    refrad1 = CFG.getfloat("Additional Forces and Perturbations","Primary Reference Radius")
    refrad2 = CFG.getfloat("Additional Forces and Perturbations","Secondary Reference Radius")
    eps1 = CFG.getfloat("Additional Forces and Perturbations","Primary Tidal Lag Angle")
    eps2 = CFG.getfloat("Additional Forces and Perturbations","Secondary Tidal Lag Angle")
    Msun = CFG.getfloat("Additional Forces and Perturbations","Sun Mass")
    
    # check initial conditions type with fahnestock flag
    if fahnestock_flag==1:
        (G,rhoA,rhoB,x0)=read_bench("systemdata_standard_MKS_units","initstate_standard_MKS_units")
        # (G,rhoA,rhoB,x0)=read_bench("systemdata_standard_MKS_units","d3_1320")
    else:
        
        # get gravity parameter - convert to kg km s units
        G=CFG.getfloat("Gravity Parameter","G")/1.e9
        
        # get densities - convert to km kg s units
        rhoA=CFG.getfloat("Body Model Definitions","Primary Density")*1.e12
        rhoB=CFG.getfloat("Body Model Definitions","Secondary Density")*1.e12
        
        # get initial conditions - convert to km kg s units
        x0=np.zeros([30])
        x0[0]=CFG.getfloat("Initial Conditions","Relative Position X")/1000.
        x0[1]=CFG.getfloat("Initial Conditions","Relative Position Y")/1000.
        x0[2]=CFG.getfloat("Initial Conditions","Relative Position Z")/1000.
        x0[3]=CFG.getfloat("Initial Conditions","Relative Velocity X")/1000.
        x0[4]=CFG.getfloat("Initial Conditions","Relative Velocity Y")/1000.
        x0[5]=CFG.getfloat("Initial Conditions","Relative Velocity Z")/1000.
        x0[6]=CFG.getfloat("Initial Conditions","Primary Angular Velocity X")
        x0[7]=CFG.getfloat("Initial Conditions","Primary Angular Velocity Y")
        x0[8]=CFG.getfloat("Initial Conditions","Primary Angular Velocity Z")
        x0[9]=CFG.getfloat("Initial Conditions","Secondary Angular Velocity X")
        x0[10]=CFG.getfloat("Initial Conditions","Secondary Angular Velocity Y")
        x0[11]=CFG.getfloat("Initial Conditions","Secondary Angular Velocity Z")
        
        if C_flag==0:
            x0[21]=CFG.getfloat("Initial Conditions","B into A (1,1)")
            x0[22]=CFG.getfloat("Initial Conditions","B into A (1,2)")
            x0[23]=CFG.getfloat("Initial Conditions","B into A (1,3)")
            x0[24]=CFG.getfloat("Initial Conditions","B into A (2,1)")
            x0[25]=CFG.getfloat("Initial Conditions","B into A (2,2)")
            x0[26]=CFG.getfloat("Initial Conditions","B into A (2,3)")
            x0[27]=CFG.getfloat("Initial Conditions","B into A (3,1)")
            x0[28]=CFG.getfloat("Initial Conditions","B into A (3,2)")
            x0[29]=CFG.getfloat("Initial Conditions","B into A (3,3)")
            C=np.reshape(x0[21:30],[3,3])

        else:# if user defines euler angles use this rotation matrix definition - MAKE SURE INPUT EULER ANGLES MATCH THEIR DEFINITIONS
            th1=CFG.getfloat("Initial Conditions","B into A Euler 1 X")
            th2=CFG.getfloat("Initial Conditions","B into A Euler 2 Y")
            th3=CFG.getfloat("Initial Conditions","B into A Euler 3 Z")
            C=np.array([[np.cos(th2)*np.cos(th3),np.sin(th1)*np.sin(th2)*np.cos(th3)+np.cos(th1)*np.sin(th3),-np.cos(th1)*np.sin(th2)*np.cos(th3)+np.sin(th1)*np.sin(th3)],\
                [-np.cos(th2)*np.sin(th3),-np.sin(th1)*np.sin(th2)*np.sin(th3)+np.cos(th1)*np.cos(th3),np.cos(th1)*np.sin(th2)*np.sin(th3)+np.sin(th1)*np.cos(th3)],\
                [np.sin(th2),-np.sin(th1)*np.cos(th2),np.cos(th1)*np.cos(th2)]]).T          
            x0[21:30]=np.reshape(C,[1,9])
            
        if Cc_flag==0:
            x0[12]=CFG.getfloat("Initial Conditions","A into N (1,1)")
            x0[13]=CFG.getfloat("Initial Conditions","A into N (1,2)")
            x0[14]=CFG.getfloat("Initial Conditions","A into N (1,3)")
            x0[15]=CFG.getfloat("Initial Conditions","A into N (2,1)")
            x0[16]=CFG.getfloat("Initial Conditions","A into N (2,2)")
            x0[17]=CFG.getfloat("Initial Conditions","A into N (2,3)")
            x0[18]=CFG.getfloat("Initial Conditions","A into N (3,1)")
            x0[19]=CFG.getfloat("Initial Conditions","A into N (3,2)")
            x0[20]=CFG.getfloat("Initial Conditions","A into N (3,3)")
        else:# if user defines euler angles use this rotation matrix definition - MAKE SURE INPUT EULER ANGLES MATCH THEIR DEFINITIONS
            th1=CFG.getfloat("Initial Conditions","A into N Euler 1 X")
            th2=CFG.getfloat("Initial Conditions","A into N Euler 2 Y")
            th3=CFG.getfloat("Initial Conditions","A into N Euler 3 Z")
            Cc=np.array([[np.cos(th2)*np.cos(th3),np.sin(th1)*np.sin(th2)*np.cos(th3)+np.cos(th1)*np.sin(th3),-np.cos(th1)*np.sin(th2)*np.cos(th3)+np.sin(th1)*np.sin(th3)],\
                [-np.cos(th2)*np.sin(th3),-np.sin(th1)*np.sin(th2)*np.sin(th3)+np.cos(th1)*np.cos(th3),np.cos(th1)*np.sin(th2)*np.sin(th3)+np.sin(th1)*np.cos(th3)],\
                [np.sin(th2),-np.sin(th1)*np.cos(th2),np.cos(th1)*np.cos(th2)]]).T  
            x0[12:21]=np.reshape(Cc,[1,9])
    
        x0[9:12]=np.dot(C,np.array([x0[9:12]]).T).T[0]
    return(G,n,nA,nB,aA,bA,cA,aB,bB,cB,a_shape,b_shape,rhoA,rhoB,t0,tf,tet_fileA,vert_fileA,tet_fileB,vert_fileB,x0,Tgen,integ,h,tol,out_freq,out_time_name,case,flyby_toggle,helio_toggle,sg_toggle,tt_toggle,Mplanet,a_hyp,e_hyp,i_hyp,RAAN_hyp,om_hyp,tau_hyp,Msolar,a_helio,e_helio,i_helio,RAAN_helio,om_helio,tau_helio,sol_rad,au_def,love1,love2,refrad1,refrad2,eps1,eps2,Msun,postProcessing)
