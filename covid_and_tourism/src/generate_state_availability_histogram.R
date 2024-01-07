library(ggplot2)
library(stringr)
library(extrafont)

generate_state_availability_histogram<- function(listings, state) { 
  availability_60_histogram <- listings %>% 
    ggplot(aes(x=availability_60)) + geom_histogram(binwidth = 3, color="black", fill="purple")  +
    labs(
      title = sprintf("Airbnb Availability in %s", state),
      x = "# of Days Airbnb Available", 
      y = "Count of Observations") +
    theme_bw() + 
    theme(text=element_text(family="Times")) +
    theme(title = element_text(size = 10)) + 
    theme(axis.title = element_text(size = 9.5))
  return(availability_60_histogram)
}