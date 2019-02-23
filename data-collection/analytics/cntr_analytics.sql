DECLARE @total_cntr_2015 INT = (SELECT count(cntr.ID) AS unique_cntrs
FROM DV.d_OOS_Contracts cntr
WHERE cntr.RefSignDate > 20150000 AND cntr.RefStage IN (3, 4))

DECLARE @total_cntr_2016 INT = (SELECT count(cntr.ID) AS unique_cntrs
FROM DV.d_OOS_Contracts cntr
WHERE cntr.RefSignDate > 20160000 AND cntr.RefStage IN (3, 4))

DECLARE @total_cntr_2017 INT = (SELECT count(cntr.ID) AS unique_cntrs
FROM DV.d_OOS_Contracts cntr
WHERE cntr.RefSignDate > 20170000 AND cntr.RefStage IN (3, 4))

DECLARE @total_cntr_2018 INT = (SELECT count(cntr.ID) AS unique_cntrs
FROM DV.d_OOS_Contracts cntr
WHERE cntr.RefSignDate > 20180000 AND cntr.RefStage IN (3, 4))

DECLARE @total_cntr_2019 INT = (SELECT count(cntr.ID) AS unique_cntrs
FROM DV.d_OOS_Contracts cntr
WHERE cntr.RefSignDate > 20190000 AND cntr.RefStage IN (3, 4))

PRINT('Кол-во контрактов с Х года')
PRINT('Кол-во контрактов с 2015 года: ' + CAST(@total_cntr_2015 AS varchar))
PRINT('Кол-во контрактов с 2016 года: ' + CAST(@total_cntr_2016 AS varchar))
PRINT('Кол-во контрактов с 2017 года: ' + CAST(@total_cntr_2017 AS varchar))
PRINT('Кол-во контрактов с 2018 года: ' + CAST(@total_cntr_2018 AS varchar))
PRINT('Кол-во контрактов с 2019 года: ' + CAST(@total_cntr_2019 AS varchar) + CHAR(13))

DECLARE @total_cntr_2015_fin INT = (SELECT count(cntr.ID) AS unique_cntrs
FROM DV.d_OOS_Contracts cntr
WHERE cntr.RefSignDate > 20150000 AND cntr.RefExecution < 20190000 AND cntr.RefStage IN (3, 4))

DECLARE @total_cntr_2016_fin INT = (SELECT count(cntr.ID) AS unique_cntrs
FROM DV.d_OOS_Contracts cntr
WHERE cntr.RefSignDate > 20160000 AND cntr.RefExecution < 20190000 AND cntr.RefStage IN (3, 4))

DECLARE @total_cntr_2017_fin INT = (SELECT count(cntr.ID) AS unique_cntrs
FROM DV.d_OOS_Contracts cntr
WHERE cntr.RefSignDate > 20170000 AND cntr.RefExecution < 20190000 AND cntr.RefStage IN (3, 4))

DECLARE @total_cntr_2018_fin INT = (SELECT count(cntr.ID) AS unique_cntrs
FROM DV.d_OOS_Contracts cntr
WHERE cntr.RefSignDate > 20180000 AND cntr.RefExecution < 20190000 AND cntr.RefStage IN (3, 4))

PRINT('Кол-во контрактов с Х года с плановой датой завершения в 2019 году')
PRINT('Кол-во контрактов с 2015 года: ' + CAST(@total_cntr_2015_fin AS varchar))
PRINT('Кол-во контрактов с 2016 года: ' + CAST(@total_cntr_2016_fin AS varchar))
PRINT('Кол-во контрактов с 2017 года: ' + CAST(@total_cntr_2017_fin AS varchar))
PRINT('Кол-во контрактов с 2018 года: ' + CAST(@total_cntr_2018_fin AS varchar) + CHAR(13))

DECLARE @total_cntr_2015_fin_not_0 INT = (SELECT count(cntr.ID) AS unique_cntrs
FROM DV.d_OOS_Contracts cntr INNER JOIN DV.f_OOS_Value val on cntr.ID = val.RefContract
WHERE cntr.RefSignDate > 20150000 AND cntr.RefExecution < 20190000 AND cntr.RefStage IN (3, 4) AND val.Price > 0)

DECLARE @total_cntr_2016_fin_not_0 INT = (SELECT count(cntr.ID) AS unique_cntrs
FROM DV.d_OOS_Contracts cntr INNER JOIN DV.f_OOS_Value val on cntr.ID = val.RefContract
WHERE cntr.RefSignDate > 20160000 AND cntr.RefExecution < 20190000 AND cntr.RefStage IN (3, 4) AND val.Price > 0)

DECLARE @total_cntr_2017_fin_not_0 INT = (SELECT count(cntr.ID) AS unique_cntrs
FROM DV.d_OOS_Contracts cntr INNER JOIN DV.f_OOS_Value val on cntr.ID = val.RefContract
WHERE cntr.RefSignDate > 20170000 AND cntr.RefExecution < 20190000 AND cntr.RefStage IN (3, 4) AND val.Price > 0)

DECLARE @total_cntr_2018_fin_not_0 INT = (SELECT count(cntr.ID) AS unique_cntrs
FROM DV.d_OOS_Contracts cntr INNER JOIN DV.f_OOS_Value val on cntr.ID = val.RefContract
WHERE cntr.RefSignDate > 20180000 AND cntr.RefExecution < 20190000 AND cntr.RefStage IN (3, 4) AND val.Price > 0)

PRINT('Кол-во контрактов с Х года с плановой датой завершения в 2019 году и положительной ценой') 
PRINT('Кол-во контрактов с 2015 года: ' + CAST(@total_cntr_2015_fin_not_0 AS varchar))
PRINT('Кол-во контрактов с 2016 года: ' + CAST(@total_cntr_2016_fin_not_0 AS varchar))
PRINT('Кол-во контрактов с 2017 года: ' + CAST(@total_cntr_2017_fin_not_0 AS varchar))
PRINT('Кол-во контрактов с 2018 года: ' + CAST(@total_cntr_2018_fin_not_0 AS varchar))