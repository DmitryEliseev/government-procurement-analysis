-- Table creation for storing federal subjects of Russia
 IF NOT EXISTS (SELECT * FROM sysobjects WHERE name = 'fed_subject_rf' AND xtype='U')
   CREATE TABLE guest.fed_subject_rf (
     ID INT NOT NULL PRIMARY KEY,
     Name VARCHAR(50)
   )
GO

IF EXISTS(SELECT * FROM sysobjects WHERE type IN ('FN', 'TF') AND name='utils_get_federal_subject')
BEGIN
  DROP FUNCTION guest.utils_get_federal_subject
END
GO

CREATE FUNCTION guest.utils_get_federal_subject (@TerID INT)

/*
This function returns federal subject ID for given territory ID
*/

RETURNS INT
AS
BEGIN
	DECLARE @FedSubjID INT = @TerID
	WHILE @FedSubjID NOT IN (SELECT ID FROM guest.fed_subject_rf)
	BEGIN
		SET @FedSubjID = (
			SELECT ter.ParentID 
			FROM DV.d_Territory_RF ter
			WHERE ter.ID = @FedSubjID
		)
	END
	RETURN @FedSubjID
END
GO