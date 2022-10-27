A<-matrix(c(0.8,0.05,0.05,0.1,
            0.1,0.8,0.05,0.05,
            0.05,0.05,0.8,0.1,
            0.1,0.05,0.05,0.8),4,4,byrow=TRUE)

Pi<-rep(0.25,4)

DNA.Markov<-function(A,Pi,t=100){
  Chain<-rep("A",t) # reserve space
  Bases<-c("A","C","G","T") # possible realization
  Chain[1]<-Bases[sample(1:4,prob=Pi,size=1)] # starting letter of the chain
  for (position in 1:t){
    row.of.A<-which(Chain[position]==Bases)
    Chain[position+1]=Bases[sample(1:4,prob=A[row.of.A,],size=1)]
  }
return(Chain)
}



DNA.Markov(A,Pi,t=100)
