IF EXISTS(SELECT * FROM sysobjects WHERE type IN ('FN', 'TF') AND name='sup_org_cntr_num')
BEGIN
  DROP FUNCTION guest.sup_org_cntr_num
END
GO

CREATE FUNCTION guest.sup_org_cntr_num (@SupID INT, @OrgID INT)

/*
Number of contracts between given supplier and customer
*/

RETURNS INT
AS
BEGIN
  DECLARE @contracts_num INT = (
    SELECT COUNT(cntr.ID)
	FROM DV.f_OOS_Value AS val
    INNER JOIN DV.d_OOS_Suppliers AS sup ON sup.ID = val.RefSupplier
    INNER JOIN DV.d_OOS_Org AS org ON org.ID = val.RefOrg
  	INNER JOIN DV.d_OOS_Contracts AS cntr ON cntr.ID = val.RefContract
		WHERE
			sup.ID = @SupID AND 
			org.ID = @OrgID AND
			cntr.RefStage IN (3, 4) AND
			cntr.RefSignDate > guest.utils_get_init_year()
	)
  RETURN @contracts_num
END
GO