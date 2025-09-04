cal_wide = read.csv("Cal_county_temp_1895_2024_Wide.csv") 

YEAR_BEGIN <- 2011 
YEAR_END <- 2024 
start <- which(cal_wide$Year==YEAR_BEGIN & cal_wide$Month == "Jan") 
end <- which(cal_wide$Year==YEAR_END & cal_wide$Month == "Dec") 

temp <- cal_wide[start:end,3:60] # 58 county 
temp <- t(temp) 

month_ids <- rep(1:12, times = (YEAR_END - YEAR_BEGIN + 1)) # numeric ids 
temp_standardized <- temp 
for (m in 1:12) { month_cols <- which(month_ids == m) 
month_data <- temp[, month_cols] 
month_mean <- rowMeans(month_data) # mean of 58 county 
month_sd <- apply(month_data, 1, sd) # sd of 58 county 

temp_standardized[, month_cols] <- sweep( sweep(month_data, 1, month_mean, "-"), # minus the mean 
                                          1, month_sd, "/" # divide the sd 
                                          ) } # save temp_standardized, not temp

#write.csv(temp_standardized, "CA_time_series.csv",row.names = F)