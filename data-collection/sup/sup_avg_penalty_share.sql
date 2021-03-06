﻿IF EXISTS(SELECT * FROM sysobjects WHERE type IN ('FN', 'TF') AND name='sup_avg_penalty_share')
BEGIN
  DROP FUNCTION guest.sup_avg_penalty_share
END
GO

CREATE FUNCTION guest.sup_avg_penalty_share (@SupID INT)

/*
Average share of penalties paid
*/

RETURNS FLOAT
AS
BEGIN
  DECLARE @avg_penalty_share FLOAT = (
    SELECT AVG(t.share)
    FROM
    (
  		SELECT DISTINCT pnl.Accrual / val.Price AS share
  		FROM DV.f_OOS_Value AS val
		  INNER JOIN DV.d_OOS_Suppliers AS sup ON sup.ID = val.RefSupplier
  		INNER JOIN DV.d_OOS_Contracts AS cntr ON cntr.ID = val.RefContract
  		INNER JOIN DV.d_OOS_Penalties AS pnl ON pnl.RefContract = cntr.ID
  		WHERE
  			sup.ID = @SupID AND 
  			cntr.RefStage IN (3, 4) AND
			  val.Price > 0 AND
			  cntr.RefSignDate > guest.utils_get_init_year()
    )t 
  )
  
  -- If no penalties have been paid by this moment
  IF @avg_penalty_share IS NULL
  BEGIN
    RETURN 0
  END
  
  RETURN ROUND(@avg_penalty_share, 3)
END
GO