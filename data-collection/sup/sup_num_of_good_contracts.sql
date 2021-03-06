IF EXISTS(SELECT * FROM sysobjects WHERE type IN ('FN', 'TF') AND name='sup_num_of_good_contracts')
BEGIN
  DROP FUNCTION guest.sup_num_of_good_contracts
END
GO

CREATE FUNCTION guest.sup_num_of_good_contracts (@SupID INT)

/*
Number of good finished contracts
*/

RETURNS INT
AS
BEGIN
  DECLARE @num_of_good_contracts INT = (
    SELECT COUNT(cntr.ID)
  	FROM DV.f_OOS_Value AS val
    INNER JOIN DV.d_OOS_Suppliers AS sup ON sup.ID = val.RefSupplier
  	INNER JOIN DV.d_OOS_Contracts AS cntr ON cntr.ID = val.RefContract
	  INNER JOIN guest.cntr_stats AS gcs ON cntr.ID = gcs.CntrID
	  WHERE
		  sup.ID = @SupID AND
		  gcs.result = 0 AND
		  cntr.RefSignDate > guest.utils_get_init_year()
  )
  RETURN @num_of_good_contracts
END
GO