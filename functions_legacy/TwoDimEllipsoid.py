from matplotlib.pyplot import plot, grid
from numpy import ones, pi, diag, cos, sin, r_, arange, sqrt, c_, sign, empty
from numpy.linalg import eigh


def TwoDimEllipsoid(Location, Square_Dispersion, Scale, PlotEigVectors, PlotSquare):
    # this def computes the location-dispersion ellipsoid
    # for details see "Risk and Asset Allocation"-Springer (2005), by A. Meucci
    # inputs:
    # 2x1 location vector (typically the expected value)
    # 2x2 scatter matrix Square_Dispersion (typically the covariance matrix)
    # a scalar Scale, that specifies the scale (radius) of the ellipsoid
    # if PlotEigVectors is 1 then the eigenvectors (=principal axes) are plotted
    # if PlotSquare is 1 then the enshrouding box is plotted. If Square_Dispersion is the covariance
    # the sides of the box represent the standard deviations of the marginals

    ######################################################################################################################################
    # compute the ellipsoid in the r plane, solution to  ((R-Location).T * Square_Dispersion**(-1) * (R-Location) ) = Scale**2
    EigenVectors, EigenValues = eigh(Square_Dispersion)
    EigenValues=diag(EigenValues)

    Angle = arange(0, 2*pi+pi/500,pi/500)
    NumSteps=len(Angle)
    Centered_Ellipse = empty((2, NumSteps))
    for i in range(NumSteps):
        z=r_[cos(Angle[i]), sin(Angle[i])]
        Centered_Ellipse[:, i] = EigenVectors.dot(diag(sqrt(EigenValues.ravel())).dot(z))
    R= Location*ones((1,NumSteps)) + Scale*Centered_Ellipse

    #################################################################################################################################
    #plots
    #################################################################################################################################
    # plot the ellipsoid
    plot(R[0,:],R[1,:], color='r',lw=2)

    #################################################################################################################################
    if PlotSquare:
        # plot a rectangle centered in Location with semisides of lengths the square roots of the diagonal of Square_Dispersion
        Dispersion=sqrt(diag(Square_Dispersion))
        Vertex_LowRight_x=Location[0]+Scale*Dispersion[0]
        Vertex_LowRight_y=Location[1]-Scale*Dispersion[1]
        Vertex_LowLeft_x=Location[0]-Scale*Dispersion[0]
        Vertex_LowLeft_y=Location[1]-Scale*Dispersion[1]
        Vertex_UpRight_x=Location[0]+Scale*Dispersion[0]
        Vertex_UpRight_y=Location[1]+Scale*Dispersion[1]
        Vertex_UpLeft_x=Location[0]-Scale*Dispersion[0]
        Vertex_UpLeft_y=Location[1]+Scale*Dispersion[1]

        Square=[Vertex_LowRight_x,Vertex_LowRight_y,
                Vertex_LowLeft_x,Vertex_LowLeft_y,
                Vertex_UpLeft_x,Vertex_UpLeft_y,
                Vertex_UpRight_x,Vertex_UpRight_y,
                Vertex_LowRight_x,Vertex_LowRight_y]
        plot(Square[:,0],Square[:,1],color='r',lw=2)

    #############################################################################################################################
    if PlotEigVectors:
        # plot eigenvectors in the r plane (centered in Location) of length the
        # square root of the eigenvalues (rescaled)

        L_1=Scale*sqrt(EigenValues[0])
        L_2=Scale*sqrt(EigenValues[1])

        # deal with reflection: matlab chooses the wrong one
        Sign= sign(EigenVectors[0,0])
        Start_x=Location[0]                               # eigenvector 1
        End_x= Location[0] + Sign*(EigenVectors[0,0]) * L_1
        Start_y=Location[1]
        End_y= Location[1] + Sign*(EigenVectors[1,0]) * L_1
        plot(c_[Start_x, End_x],c_[Start_y, End_y], color='r',lw=2)

        Start_x=Location[0]                               # eigenvector 2
        End_x= Location[0] + (EigenVectors[0,1]*L_2)
        Start_y=Location[1]
        End_y= Location[1] + (EigenVectors[1,1]*L_2)
        plot(c_[Start_x, End_x],c_[Start_y, End_y], color='r',lw=2)

    #############################################################################################################################
    grid(True)
