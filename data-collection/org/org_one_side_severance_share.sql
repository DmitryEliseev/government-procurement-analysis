IF EXISTS(SELECT * FROM sysobjects WHERE type IN ('FN', 'TF') AND name='org_one_side_severance_share')
BEGIN
  DROP FUNCTION guest.org_one_side_severance_share
END
GO

CREATE FUNCTION guest.org_one_side_severance_share (@OrgID INT, @NumOfCntr INT)

/*
Skandalousness of customer: share of contracts which were terminated unilaterally
*/

RETURNS FLOAT
AS
BEGIN
  DECLARE @num_of_bad_contracts INT = (
  	SELECT COUNT(*)
  	FROM
  	(
  		SELECT DISTINCT cntr.ID
  		FROM DV.f_OOS_Value AS val
  		INNER JOIN DV.d_OOS_Org AS org ON org.ID = val.RefOrg
  		INNER JOIN DV.d_OOS_Contracts AS cntr ON cntr.ID = val.RefContract
  		INNER JOIN DV.d_OOS_ClosContracts As cntrCls ON cntrCls.RefContract = cntr.ID
  		INNER JOIN DV.d_OOS_TerminReason AS trmn ON trmn.ID = cntrCls.RefTerminReason
  		WHERE 
  			trmn.Code IN (8326974, 8361022, 8724082) AND
  			org.ID = @OrgID AND
  			cntr.RefSignDate > guest.utils_get_init_year()
  	)t
  )
  
  -- If customer has no contracts by this moment
  IF @NumOfCntr = 0
  BEGIN
    RETURN 0
  END
  
  RETURN ROUND(@num_of_bad_contracts / @NumOfCntr, 3)
END
GO