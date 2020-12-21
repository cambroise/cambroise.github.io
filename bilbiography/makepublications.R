
library(bibtex)
library(hash)
library(stringr)
##############################        
# Just input bibentry to modify or add 
#   in the bibfile
#############################@
bibfile.orig<-"google.bib"
x <- readLines(bibfile.orig)
y <- str_remove_all(x,"\\\\'" )
y <- str_remove_all(y,"\\\\`" )
y <- str_remove_all(y,'\\\\"' )
y <- str_replace_all(y, '\\\\&','&' )
#y <- str_remove_all(y, '\\H' )


cat(y, file="google-clean.bib", sep="\n")
bibfile<-"google-clean.bib"
#bibfile<-"test.bib"

mypubs<-read.bib(bibfile)
dir.create('../content/publication')

process.bibentry<-function(entry,path="../content/publication/"){
  # Get the key, associated dir and cite.bib
   key<-entry$key
   dir.create(paste(path,key,sep=""))
   write.bib(entry,file = paste(path,key,"/","cite.bib",sep=""))
   
   # create the index.md
   filename <- file.path(paste(path,key,"/","index.md",sep=""))
   write("---", filename)
   if (!is.null(entry$author)) {
     write("authors: ",file=filename,append=T)
     authors<-unlist(lapply(entry$author,
                            function(x) 
                              paste(x$family," ",
                                    str_sub(x$given[1],1,1)," ",sep="")))
     for (author in authors) write(paste("- ",author,sep=""),file=filename, append=T)
   }
   
   if (!is.null(entry$year)) {
     
     write(paste("date: ",entry$year,"-01-01",sep=""),file=filename,append=T)
   }
   
   if (!is.null(entry$title)) {
     write(paste("title:",'"',entry$title,'"'),file=filename,append=T)
   }
   
   if (!is.null(entry$doi)) {
     write(paste("doi:",entry$doi),file=filename,append=T)
   }
   
   if (!is.null(entry$journal)) {
     write(paste("publication:",'"',entry$journal,'"'),file=filename,append=T)
     if  ((strsplit(entry$journal,' ')[[1]][1]=="arXiv")| (strsplit(entry$journal,' ')[[1]][1]=="bioRxiv")){
       if (abs(as.numeric(substring(date(),21,24))-as.numeric(entry$year))<3){
       write("featured: true",filename,append=T)}
     }
          }
   bibtype2pubtype<-hash(c("Article",
                     "Book",
                     "InProceedings",
                     "Proceedings",
                     "Conference",
                     "MastersThesis",
                     "PhdThesis",
                     "Manual",
                     "TechReport",
                     "InCollection",
                     "InBook",
                     "Misc"), 
      c(2,5,1,1,1,3,3,4,4,6,6,0))
   
   write("publication_types:",filename,append=T)
   write(paste("-",bibtype2pubtype[[entry$bibtype]]),filename,append=T)       
   
   write("url_pdf: ''",filename,append=T)
   
   
   write("---", filename,append=T)
   
}

lapply(mypubs,process.bibentry)
