IF EXISTS(SELECT * FROM sysobjects WHERE type IN ('FN', 'TF') AND name='target')
BEGIN
  DROP FUNCTION guest.target
END
GO

CREATE FUNCTION guest.target (@CntrID INT)

/*
This function defines status of contract: weather it is bad or not.
In previous work (bachelor theses, 2018) contract was good, if it meets following requirements:
1. Contract was finished OR
2. Contract was terminated by mutual consent AND by this moment contract was executed by more than 60%

In this work definition of good contract will be more strict. Contract must be finished to be regarded as good.
Second option is no more valid. In all other cases contract is regarded as bad.
*/

RETURNS INT
AS
BEGIN
  DECLARE @Target INT = (
    SELECT 'target' =
  	  CASE WHEN t.Code = 0 THEN 0 ELSE 1
  	  END
  	FROM
  	(
      -- Выбор последней записи для контрактов с несколькими стадиями,
      -- проблемы у которых начались не с первой стадии
  	  SELECT TOP(1)
      cntr.ID, trmn.Code, val.Price, sum(clsCntr.FactPaid) AS Done
  	  FROM DV.f_OOS_Value as val
  	  INNER JOIN DV.d_OOS_Contracts AS cntr ON cntr.ID = val.RefContract
  	  INNER JOIN DV.d_OOS_ClosContracts AS clsCntr ON clsCntr.RefContract = cntr.ID
  	  INNER JOIN DV.d_OOS_TerminReason AS trmn ON trmn.ID = clsCntr.RefTerminReason
  	  WHERE cntr.ID = @CntrID
  	  GROUP BY cntr.ID, trmn.Code, val.Price
      ORDER BY trmn.Code DESC
  	)t
  )
  RETURN @Target
END
GO