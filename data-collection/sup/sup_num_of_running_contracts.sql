IF EXISTS(SELECT * FROM sysobjects WHERE type IN ('FN', 'TF') AND name='sup_num_of_running_contracts')
BEGIN
  DROP FUNCTION guest.sup_num_of_running_contracts
END
GO

CREATE FUNCTION guest.sup_num_of_running_contracts (@SupID INT)

/*
Number of contracts which are currently in process of execution
*/

RETURNS FLOAT
AS
BEGIN
  DECLARE @num_of_all_finished_contracts FLOAT = (
  	SELECT COUNT(cntr.ID)
  	FROM DV.f_OOS_Value AS val
  	INNER JOIN DV.d_OOS_Suppliers AS sup ON sup.ID = val.RefSupplier
  	INNER JOIN DV.d_OOS_Contracts AS cntr ON cntr.ID = val.RefContract
  	WHERE 
  		sup.ID = @SupID AND 
  		cntr.RefStage = 2 AND
  		cntr.RefSignDate > guest.utils_get_init_year()
  )
  RETURN @num_of_all_finished_contracts
END
GO