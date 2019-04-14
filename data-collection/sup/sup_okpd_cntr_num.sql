IF EXISTS(SELECT * FROM sysobjects WHERE type IN ('FN', 'TF') AND name='sup_okpd_cntr_num')
BEGIN
  DROP FUNCTION guest.sup_okpd_cntr_num
END
GO

CREATE FUNCTION guest.sup_okpd_cntr_num (@SupID INT, @OkpdID INT)

/*
Number of contract for given OKPD. OKPD is abbreviation for
russian classification of products by economic activity.
Link: https://tender-rus.ru/okpd
*/

RETURNS INT
AS
BEGIN
  DECLARE @cur_okpd_contracts_num INT = (
    SELECT COUNT(*)
    FROM
    (
      SELECT DISTINCT cntr.ID
      FROM DV.f_OOS_Product AS prod
      INNER JOIN DV.d_OOS_Suppliers AS sup ON sup.ID = prod.RefSupplier
      INNER JOIN DV.d_OOS_Contracts AS cntr ON cntr.ID = prod.RefContract
      INNER JOIN DV.d_OOS_Products AS prods ON prods.ID = prod.RefProduct
      WHERE 
        sup.ID = @SupID AND 
        cntr.RefStage in (3, 4) AND
		    prods.RefOKPD2 = @OkpdID AND
		    cntr.RefSignDate > guest.utils_get_init_year()
    )t
  ) 
  RETURN @cur_okpd_contracts_num
END
GO