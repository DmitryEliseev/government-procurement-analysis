IF EXISTS(SELECT * FROM sysobjects WHERE type IN ('FN', 'TF') AND name='utils_get_init_year')
BEGIN
  DROP FUNCTION guest.utils_get_init_year
END
GO

CREATE FUNCTION guest.utils_get_init_year ()

/*
This is starting year which is used for counting metrics, e.g.
number of good contracts for supplier, average contract price for customer and etc.

The environment of government procurement in Russia is constantly evolving, that's why
it was decided not to hardcode this constant.
*/

RETURNS INT
AS
BEGIN
  RETURN 20160000
END
GO