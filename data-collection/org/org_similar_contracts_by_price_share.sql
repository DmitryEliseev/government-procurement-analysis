IF EXISTS(SELECT * FROM sysobjects WHERE type IN ('FN', 'TF') AND name='org_similar_contracts_by_price_share')
BEGIN
  DROP FUNCTION guest.org_similar_contracts_by_price_share
END
GO

CREATE FUNCTION guest.org_similar_contracts_by_price_share (@OrgID INT, @NumOfCntr INT, @CntrPrice BIGINT)

/*
Number of finished contracts with price lying in +-20% range from price of current contract
*/

RETURNS FLOAT
AS
BEGIN
  DECLARE @num_of_similar_contracts_by_price FLOAT = (
	SELECT COUNT(cntr.ID)
  	FROM DV.f_OOS_Value AS val
  	INNER JOIN DV.d_OOS_Org AS org ON org.ID = val.RefOrg
  	INNER JOIN DV.d_OOS_Contracts AS cntr ON cntr.ID = val.RefContract
  	WHERE 
		org.ID = @OrgID AND 
  		cntr.RefStage IN (3, 4) AND 
  		ABS(val.Price - @CntrPrice) <= 0.2 * @CntrPrice AND
  		cntr.RefSignDate > guest.utils_get_init_year()
  )
  
  -- If customer hasn't had contracts by this moment
  IF @NumOfCntr = 0
  BEGIN
    RETURN 0
  END
  
  RETURN ROUND(@num_of_similar_contracts_by_price / @NumOfCntr, 3)
END
GO