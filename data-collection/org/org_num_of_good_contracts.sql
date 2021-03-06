IF EXISTS(SELECT * FROM sysobjects WHERE type IN ('FN', 'TF') AND name='org_num_of_good_contracts')
BEGIN
  DROP FUNCTION guest.org_num_of_good_contracts
END
GO

CREATE FUNCTION guest.org_num_of_good_contracts (@OrgID INT)

/*
Number of good finished contracts
*/

RETURNS INT
AS
BEGIN
  DECLARE @num_of_good_contracts INT = (
    SELECT COUNT(cntr.ID)
    FROM DV.f_OOS_Value AS val
    INNER JOIN DV.d_OOS_Org AS org ON org.ID = val.RefOrg
    INNER JOIN DV.d_OOS_Contracts AS cntr ON cntr.ID = val.RefContract
	  INNER JOIN guest.cntr_stats AS gcs ON cntr.ID = gcs.CntrID
    WHERE 
  		org.ID = @OrgID AND 
		  gcs.result = 0 AND
		  cntr.RefSignDate > guest.utils_get_init_year()
  )
  RETURN @num_of_good_contracts
END
GO