format_date <- function(var) { 
  return(as.Date(strtoi(var), origin = "1899-12-30"))
}